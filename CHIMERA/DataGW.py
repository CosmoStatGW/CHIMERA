from abc import ABC, abstractmethod
import numpy as np
import h5py, os, json, requests
import logging
log = logging.getLogger(__name__)

from CHIMERA.utils import angles

__all__ = [
    "DataGWMock",
    "DataLVK",
]


class DataGW(ABC):
    # Generate docstring for abstract class
    """Abstract class to handle GW data operations."""
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def load():
        pass
    
    def subsample_posteriors(self, N_sample):

        for name, posteriors in self.data.items():
            N_original = len(posteriors[list(posteriors.keys())[0]])

            subsample  = np.random.choice(N_original, N_sample, replace=False)

            self.data[name] = {k: v[subsample] for k,v in posteriors.items()}

    def medians(self):
        """Return the median values of the parameters."""
        return {k:np.median(v, axis=1) for k,v in self.data_array.items()}



class DataGWMock(DataGW):
    """Class to load a mock GW catalog.
    
    Args:   
        dir_catalog (str): path to the mock GW catalog file.
    """
    
    def __init__(self, dir_catalog):
        self._file = dir_catalog


    def load(self, 
             Nevents   = None,
             Nsamples  = None,
             keys_load = ["m1det", "m2det", "chi1z", "phi", "dL", "theta", "chi2z", "iota"],
             ):
        
        """Load mock GW catalog calculating (RA,DEC) values.

        Args:
            Nevents (int, optional): number of events to load. Defaults to all events.
            Nsamples (int, optional): number of samples to load for each event. Defaults to all samples.
            keys_load (list, optional): list of keys to be loaded. Defaults to ["m1det", "m2det", "phi", "dL", "theta"].

        Returns:
            dict: dictionary of arrays containing the mock GW data.
        """

        data = {}

        log.info("Loading mock GW catalog...")

        with h5py.File(self._file, 'r') as f:

            Nevents_max, Nsamples_max = f["posteriors"][keys_load[0]].shape

            if Nevents is None: Nevents = Nevents_max
            if Nsamples is None: Nsamples = Nsamples_max

            for f_key in keys_load:
                if f_key not in f['posteriors'].keys(): 
                    log.warning(" > key {:s} not available".format(f_key))
                else:
                    data[f_key] = np.array(f["posteriors"][f_key])[:Nevents, :Nsamples]

        self.Nevents  = Nevents
        self.Nsamples = Nsamples

        log.info(" > loaded {:d} events with {:d} posterior samples each".format(self.Nevents, self.Nsamples))
        log.info(" > converting (theta,phi) to (RA,DEC) [rad] and dL [Gpc]")

        ra, dec      = angles.ra_dec_from_th_phi(data["theta"], data["phi"])
        data["ra"]   = ra
        data["dec"]  = dec

        self.data_array = data

        return data
    



class DataLVK(DataGW):
    """Class to load the LVK GW catalog.
    
    Args:   
        dir_catalog (str): path to the mock GW catalog file.
    """

    def __init__(self, dir_catalog, **kwargs):

        super().__init__(**kwargs)

        self.dir_file   = dir_catalog 
        self.table      = self._get_LVK_table(os.path.join(dir_catalog, "GWTC-confident.json"))
        self.data       = None
        self.data_array = None


    def load(self, event_list, Nsamples=None, exact_name=False, table_name=None, remap_keys=None):
        """Load a list of events from the LVK catalog.

        Args:
            event_list (list): list of event names
            Nsamples (int, optional): number of samples to load for each event. Defaults to all samples.
            exact_name (bool, optional): whether to use the exact name of the event. Defaults to False.
            table_name (str, optional): name of the table to load. Defaults to None.
            remap_keys (dict, optional): dictionary to remap the keys. Defaults to None.

        Returns:    
            dict: dictionary of arrays containing the GW data.
        """

        self.names      = np.atleast_1d(event_list)
        self.exact_name = exact_name
        self.table_name = table_name
        self.remap_keys = remap_keys
        self.data       = {n:self.load_event(n) for n in self.names}

        if Nsamples is not None:
            self.subsample_posteriors(Nsamples)

        # convert to dict of parameters with arrays of shape (Nevent, Nsamples)
        param_names     = list(self.data[self.names[0]].keys())

        self.data_array = {k : np.array([self.data[e][k] for e in self.names]) for k in param_names}

        return self.data_array



    def load_event(self, name, which_spins=None):
        """Load a single event from the LVK catalog.

        Args:
            name (str): name of the event
            which_spins (str, optional): which spins to load. Defaults to None.

        Returns:
            dict: dictionary of arrays containing the GW data.
        """

        keysCHIMERA = ["m1det", "m2det", "dL", "ra", "dec"]

        obs_run, _  = self._get_run_properties(name)

        log.info(f"Loading {name} from {obs_run}...")

        if obs_run == "O1O2":
            keysLVK    =  ['m1_detector_frame_Msun', 'm2_detector_frame_Msun', 'luminosity_distance_Mpc', 
                           'right_ascension', 'declination']
            name_ext   = name + ".hdf5" if self.exact_name else name + "_GWTC-1.hdf5"
            print(name_ext)
            dir_file   = os.path.join(self.dir_file, obs_run, name_ext)
            
            with h5py.File(dir_file, 'r') as f:
                table_name = 'Overall_posterior' if self.table_name is None else self.table_name
                posterior_samples = f[table_name]

                if self.remap_keys is not None:
                    event = {k: posterior_samples[v] for k,v in self.remap_keys.items()}   
                else:
                    event = {keysCHIMERA[i]: posterior_samples[k] for i,k in enumerate(keysLVK)}


            if which_spins is None:
                pass
            elif which_spins=='s1s2':
                event['spin1'] = posterior_samples['spin1']
                event['spin2'] = posterior_samples['spin2']     
            elif which_spins=='chiEff':
                s1, s2          = posterior_samples['spin1'], posterior_samples['spin2']
                cost1, cost2    = posterior_samples['costilt1'], posterior_samples['costilt2']
                sint1, sint2    = np.sqrt(1-cost1**2), np.sqrt(1-cost2**2)
                chi1z, chi2z    = s1*cost1, s2*cost2
                q               = self.data["m2det"]/self.data["m1det"]
                chiEff          = (chi1z+q*chi2z)/(1+q)
                chiP            = np.max( np.array([s1*sint1, (4*q+3)/(4+3*q)*q*s2*sint2]) , axis=0 )
                event['chiEff'] = chiEff
                event['chiP']   = chiP
            else:
                raise NotImplementedError()
        
        elif obs_run == "O3a":
            keysLVK    = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec']
            name_ext   = name + ".hdf5"  if self.exact_name else name + '.h5'
            # dir_file   = os.path.join(self.dir_file, obs_run, name_ext) # fixme
            dir_file   = os.path.join(self.dir_file, name_ext)
            # print(dir_file)
            with h5py.File(dir_file, 'r') as f:
                
                if self.table_name is None:
                    try:
                        posterior_samples = f['C01:IMRPhenomD']['posterior_samples']
                    except:
                        raise ValueError(f"key not found in file, available: {list(f.keys())}")
                
                else:
                    posterior_samples = f[self.table_name]

                if self.remap_keys is not None:
                    event = {k: np.array(posterior_samples[v]) for k,v in self.remap_keys.items()}   
                else:
                    event = {keysCHIMERA[i]: np.array(posterior_samples[k]) for i,k in enumerate(keysLVK)}

            if which_spins is None:
                pass    
            elif which_spins=='chiEff':
                event['chiEff'] = posterior_samples['chiEff']
                event['chiP']   = posterior_samples['chiP']
            else:
                raise NotImplementedError()

        elif obs_run == "O3b":
            keysLVK = ['mass_1', 'mass_2', 'luminosity_distance', 'ra', 'dec']
            name_ext   = name_ext   = name if self.exact_name else 'IGWN-GWTC3p0-v1-' + name + '_PEDataRelease_mixed_nocosmo.h5'
            dir_file   = os.path.join(self.dir_file, obs_run, name_ext)

            with h5py.File(dir_file, 'r') as f:
                try:
                    posterior_samples = f['C01:IMRPhenomD']['posterior_samples']
                except:
                    raise ValueError(f"key not found in file, available: {list(f.keys())}")
                event = {keysCHIMERA[i]: posterior_samples[k] for i,k in enumerate(keysLVK)}

            if which_spins is None:
                pass
            elif which_spins=='chiEff':
                event['chiEff'] = posterior_samples['chiEff']
                event['chiP']   = posterior_samples['chiP']
            else:
                raise NotImplementedError()
        
        return event


    def _get_LVK_table(self, filename):
        # Check if the file is in the directory 
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                data = json.load(f)["events"]
        else:
            print(f"File '{filename}' not found in directory, downloading from gwosc.org...")

            response = requests.get("https://gwosc.org/eventapi/json/GWTC/")

            # Check if the request was successful
            if response.status_code == 200:
                with open(filename, 'w') as outfile:
                    outfile.write(response.text)
                data = json.loads(response.content)["events"]
            else:
                print(f"Request failed with status code {response.status_code}")
                return None
            
        return data


    def get_key(self, name):
        vers_keys = [key for key in self.table.keys() if key.startswith(name)]

        if vers_keys == []:
            raise Exception(f"Event {name} not found in the LVK table.")
        
        return max(vers_keys, key=lambda s: int(s.split('-v')[-1]))


    def _get_run_properties(self, name):

        gpstime = self.table[self.get_key(name)]["GPS"]

        if 1125964817 < gpstime < 1187827217:
            # The first observing run (O1) ran from September 12th, 2015 to January 19th, 2016 --> 129 days
            # From https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.041015: 
            # after data quality flags, the remaining coincident analysis time in O1 is 48.3 days with GSTLal analysis, 46.1 with pycbc
            # The second observing run (O2) ran from November 30th, 2016 to August 25th, 2017 --> 267 days
            # During the O2 run the duty cycles were 62% for LIGO Hanford and 61% for LIGO Livingston, 
            # so that two detectors were in observing mode 46.4% of the time and at least one detector 
            # was in observing mode 75.6% of the time.
            # From https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031040 :
            # During O2, the individual LIGO detectors had duty factors of approximately 60% with a LIGO 
            # network duty factor of about 45%. Times with significant instrumental disturbances are flagged and removed, 
            # resulting in about 118 days of data suitable for coincident analysis
            return "O1O2", (48.3+118)/365.25  # yrs
        
        elif 1238112018 < gpstime < 1254096017:
            # O3a data is taken between 1 April 2019 15:00 UTC and 1 October 2019 15:00 UTC.
            # The duty cycle for the three detectors was 76% (139.5 days) for Virgo, 
            # 71% (130.3 days) for LIGO Hanford, and 76% (138.5 days) for LIGO Livingston. 
            # With these duty cycles, the full 3-detector network was in observing mode 
            # for 44.5% of the time (81.4 days). 
            # Moreover, for 96.9% of the time (177.3 days) at least one detector was
            # observing and for 81.9% (149.9 days) at least two detectors were observing.
            return "O3a", 183.3/365.25  # yrs
        
        elif 1256601618 < gpstime < 1269475217:
            # O3b dates: 1st November 2019 15:00 UTC (GPS 1256655618) to 27th March 2020 17:00 UTC (GPS 1269363618)
            # (147.1) 148 days in total. 142.0 days with at least one detector for O3b      
            # Second half of the third observing run (O3b) between 1 November 2019, 15:00 UTC and 27 March 2020, 17:00 UTC
            # for 96.6% of the time (142.0 days) at least one interferometer was observing,
            # while for 85.3% (125.5 days) at least two interferometers were observing
            return "O3b", 147.1/365.25  # yrs

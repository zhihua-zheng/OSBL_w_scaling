Description of Lagrangian float data 

===== Labrador Sea 1997/1998 (Steffen & D'Asaro 2002)
General note: floats were programmed for two month missions. They were 
also programmed to go through a non-Lagrangian autoballast sequence for a 
week at the start of their missions.

In 1997 a problem with the weight attachment resulted in gradual failures 
as weights detached from the floats.

In 1998 the floats were programmed to perform a profile... the last
few days of these records are not Lagrangian.

Float calibrations show temperature good to .001C, except for float 21 
which had a bad calibration file, so temp data from that float should 
probably be ignored.

Floats had RAFOS tracking for position every 4 hours.


===== OCS Papa 2011/2012, Lake Washington 2011/2012 (D'Asaro et al. 2014)
 B: piston position
 FM: ?
 GPS: time (mtime), latitude (lat), longitude (lon), and number of 
      satellites (nsats) used in the float surfing and data transmission
 hr: hours since deployment
 Humidity: internal hull humidity?
 Int_P: internal hull pressure (half Patm in calibration)
 lat: interpolated latitude
 lon: interpolated longitude
 Mtime: matlab date number
 P: pressure on the float hull [dbar]
 PP: pressure from the CTD sensor [dbar]
 Pav: lagged pressure? [dbar]
 S: in-situ salinity [psu]
 Sig0: surface referenced potential density [kg/m^3]
 T: in-situ temperature [C]
 Th: potential temperature [C]
 Volts: battery voltages [V] 
 yd: yearday since deployment

 mode: different modes of float
  0: (down) downward profiling
  1: (settle) ballast
  2: (up) upward profiling
  3: (drift) adjusting to the density last measured within 
             the selected middle part of boundary layers
  4: (drift) isopycnal tracking?
  5: (drift) adjusting to the density currently measured  
  safemode?
  drogue
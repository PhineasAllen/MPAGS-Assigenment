# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:37:40 2023

@author: Student
"""
# Import necessary libraries
import skyfield
from datetime import datetime
from skyfield.api import load, wgs84, EarthSatellite
import numpy as np
import astropy
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import astropy.coordinates as coord
from astropy.time import Time
import skyproj
import astropy
from astropy.table import QTable
import os 
import math
    
# Define some initial parameters and constants
# These values might represent characteristics of the camera detector and other simulation parameters
DectX = int(9576/4)
DectY = int(6388/4)
PixScale = 1.25*4
Seeing = 2  # arcsecs
satmag = 8
std = Seeing/PixScale
shape = [DectX,DectY]
RefMag = 8
RefCount = 10000
ExposureTime = 0.2
std = 1.5

# Define location and time parameters
stations_url, lat, long, elev, hrs, mins, secs, year, month, day = "1710.txt", 28.757358708253182, -17.88495998043063, 2400, 1, 60, 60, 2023, 10, 9

# Function to calculate the number of stars based on their magnitudes
def GetStarCounts(RefMag, RefCount, StarMag): 
    return RefCount * (100 ** ((RefMag - StarMag) / 5))


def MakeStarTrailTable(StarMag, StarXPos, StarYPos, RefMag, RefCount, ExposureTime, std): 
    # Initialize empty lists for amplitude and standard deviation
    amp = [] 
    xstd = []
    
    # Calculate total star counts based on reference magnitude and count
    TotalStarCounts = GetStarCounts(RefMag, RefCount, StarMag)
    PerPixelStarFlux = TotalStarCounts / len(StarXPos)  # Calculate flux per pixel
    
    # Populate lists for amplitude and standard deviation for each star
    for v in range(0, len(StarXPos)):
        amp = amp + [PerPixelStarFlux]  # Add per-pixel flux to amp list
        xstd = xstd + [std]  # Add standard deviation to xstd list
    
    ystd = xstd  # Set ystd same as xstd
    
    # Return lists containing amplitude, star X positions, star Y positions, and standard deviations
    return amp, StarXPos, StarYPos, xstd, ystd

def rotate_point(x, y, cx, cy, theta):
    # Translate the point to the origin by subtracting the central point coordinates
    translated_x = x - cx
    translated_y = y - cy

    # Perform the rotation using the rotation matrix formula
    # Calculate new coordinates after rotation
    rotated_x = translated_x * math.cos(theta) - translated_y * math.sin(theta)
    rotated_y = translated_x * math.sin(theta) + translated_y * math.cos(theta)
    
    # Translate the point back to its original position by adding the central point coordinates
    new_x = rotated_x + cx
    new_y = rotated_y + cy
    
    # Return the new coordinates after rotation
    return new_x, new_y


# Function to calculate angular separation between two celestial coordinates in degrees
def Deg_On_Sky_Astropy(SkyCoord1, SkyCoord2):
    Sep = SkyCoord1.separation(SkyCoord2)  # Calculate angular separation
    return Sep.deg  # Return separation in degrees

# Function to convert angular separations to pixel separations
def Pix_Sep(SkyCoord1, SkyCoord2, pix):
    # Get angular separation using previously defined function
    Sep = Deg_On_Sky_Astropy(SkyCoord1, SkyCoord2)
    
    # Calculate position angle and pixel separations
    Vector = 2 * np.pi - astropy.coordinates.position_angle(SkyCoord1.ra.rad, SkyCoord1.dec.rad, SkyCoord2.ra.rad, SkyCoord2.dec.rad).rad
    XSep = math.cos(Vector) * Sep
    YSep = math.sin(Vector) * Sep
    
    # Convert angular separation to pixel separation using the specified pixel scale (pix)
    XPixSep = YSep * 3600 / pix  # Convert Y separation to pixel units
    YPixSep = XSep * 3600 / pix  # Convert X separation to pixel units
    
    return XPixSep, YPixSep  # Return X and Y pixel separations

def Make_Arc_and_Move(cenx, ceny, x, y, theta, AltDiff, AzDiff, x2, y2, arc_x, arc_y, pix, DectX, DectY):
    # Convert angle to radians for trigonometric functions
    angle_in_radians = np.radians(theta)
    
    # Initialize variables and parameters
    count = 1
    res = 50
    DetectorCenter = [DectX/2, DectY/2]  # Calculate detector center
    SatPosition = SkyCoord(ra=cenx, dec=ceny, unit=(u.deg, u.deg))  # Satellite position
    StarPosition = SkyCoord(ra=x, dec=y, unit=(u.deg, u.deg))  # Star position
    XPixSep, YPixSep = Pix_Sep(SatPosition, StarPosition, pix)  # Get pixel separations
    StarDetectorPosition = [DetectorCenter[0] + XPixSep, DetectorCenter[1] + YPixSep]  # Star position on detector
    StartingDetectorPosition = [DetectorCenter[0] + XPixSep, DetectorCenter[1] + YPixSep]  # Starting position
    
    # Initialize lists for positions
    arc_x, arc_y = [], []
    LatestSatPosition = SkyCoord(ra=cenx + AzDiff, dec=ceny + AltDiff, unit=(u.deg, u.deg))  # Latest satellite position
    XDrift, YDrift = Pix_Sep(SatPosition, LatestSatPosition, pix)  # Calculate pixel drift

    JustDriftX = []
    JustDriftY = []
    
    # Loop to create the arc and move the star
    for f in range(0, res):
        # Rotate star position
        rotated_x, rotated_y = rotate_point(StarDetectorPosition[0], StarDetectorPosition[1], DetectorCenter[0], DetectorCenter[1], angle_in_radians / res)
        StarDetectorPosition = [rotated_x - XDrift / res, rotated_y - YDrift / res]  # Update rotated position
        JustDriftX = JustDriftX + [StartingDetectorPosition[0] - count * XDrift / res]  # Calculate drift in X
        JustDriftY = JustDriftY + [StartingDetectorPosition[1] - count * YDrift / res]  # Calculate drift in Y
        arc_x = arc_x + [StarDetectorPosition[0]]  # Store X position
        arc_y = arc_y + [StarDetectorPosition[1]]  # Store Y position
        count = count + 1  # Increment count
    
    return np.array(arc_x), np.array(arc_y), JustDriftX, JustDriftY


# Define the URL for TLE (Two-Line Element) data
# Actual TLE file paths


def GetSats(stations_url, lat, long, elev, hrs, mins, secs, year, month, day):
    # Load TLE (Two-Line Element) data from the specified URL
    satellites = load.tle_file(stations_url)
    print('Loaded', len(satellites), 'satellites')
    
    # Define the Earth location using latitude, longitude, and elevation
    la_palma = wgs84.latlon(lat, long, elevation_m=elev)
    ts = load.timescale()
    t = ts.now()
    time = []
    
    # Loop through each satellite in the loaded TLE data
    for satellite in satellites:
        alts, azs, dis = [], [], []  # Lists to store altitude, azimuth, and distance data
        ras, decs = [], []  # Lists to store right ascension and declination data
        
        # Calculate the difference between the satellite and the Earth location
        difference = satellite - la_palma
        
        # Iterate through specified hours, minutes, and seconds
        for k in range(1, 1 + hrs):
            for j in range(38, 39):
                for i in range(0, secs):
                    for f in range(0, 800, 200):
                        time = time + [Time(datetime(year, month, day, k, j, i, f), scale='utc')]  # Store time
                        
                        # Calculate satellite's topocentric coordinates at the given time
                        t = ts.utc(year, month, day, k, j, i)
                        topocentric = difference.at(t)
                        alt, az, distance = topocentric.altaz()  # Get altitude, azimuth, and distance
                        ra, dec, distance = topocentric.radec()  # Get right ascension and declination
        
                        # Store data in respective lists
                        alts = alts + [alt.degrees]
                        azs = azs + [az.degrees]
                        ras = ras + [ra._degrees]
                        decs = decs + [dec._degrees]
                    
        tk, zk = [], []  # Lists for altitude and azimuth data
        td, zd = [], []  # Lists for altitude and azimuth differences
        rk, dk = [], []  # Lists for right ascension and declination data
        timek = []  # List for time data
        
        # Interpolate for sub-second exposures
        for i in range(0, len(time) - 4, 4):
            altdiff = -alts[i] + alts[i + 4]
            azdiff = -azs[i] + azs[i + 4]
            rasdiff = -ras[i] + ras[i + 4]
            decsdiff = -decs[i] + decs[i + 4]
            
            if abs(azdiff) < 10:
                for j in range(1, 4):
                    alts[i + j] = alts[i + j - 1] + altdiff / 4
                    azs[i + j] = azs[i + j - 1] + azdiff / 4
                    decs[i + j] = decs[i + j - 1] + decsdiff / 4
                    ras[i + j] = ras[i + j - 1] + rasdiff / 4
        
        # Store data for valid altitude and azimuth conditions
        for i in range(1, len(alts) - 1):
            if abs(azs[i] - azs[i + 1]) < 90 and abs(azs[i] - azs[i - 1]) < 90 and alts[i] > 0:
                tk = tk + [alts[i]]
                zk = zk + [azs[i]]
                td = td + [alts[i + 1] - alts[i]]
                zd = zd + [azs[i + 1] - azs[i]]
                rk = rk + [ras[i]]
                dk = dk + [decs[i]]
                timek = timek + [time[i]]
            else:
                timek = timek + [time[i]]
                tk = tk + [np.nan]
                zk = zk + [np.nan]
        
        # Print satellite data if mean altitude is greater than 30 degrees
        if np.mean(alts) > 30:
            print(satellite)
    
    return satellite, zk, tk, zd, td, rk, dk, timek  # Return satellite and associated data


def GetStars_Plot(lat, long, elev, rk, dk, timek, zk, tk):
    adder = 0
    keptresiduals = []  # Initialize lists to store residual values
    for i in range(0, len(timek)):
        # Set up the observing location
        observing_location = EarthLocation(lat=str(lat), lon=str(long), height=elev * u.m)
        star_alts, star_azs = [], []  # Lists to store star altitudes and azimuths
        pltx, plty = [], []  # Lists to store plot x and y coordinates
        amps, xstars, ystars, xstds, ystds = [], [], [], [], []  # Initialize lists for star data

        v = Vizier(columns=['_RAJ2000', '_DEJ2000', 'Hpmag', 'Vmag'], column_filters={"Vmag": "<11"})
        v.ROW_LIMIT = 1000
        
        # Query Vizier catalog for star data around the satellite's position
        result = v.query_region(SkyCoord(ra=rk[i + 1], dec=dk[i + 1], unit=(u.deg, u.deg), frame='icrs'),
                                radius=Angle(2, "deg"), catalog=["NOMAD"])

        # Loop through the stars in the Vizier catalog result
        for val in result[0]:
            # Get the observing time for the current data point
            observing_time = timek[i]

            # Define an AltAz coordinate system for the observing location and time
            aa = AltAz(location=observing_location, obstime=observing_time)

            # Create a SkyCoord object for the star's RA and DEC in the ICRS coordinate system
            coordr = SkyCoord(ra=val[0], dec=val[1], unit=(u.deg, u.deg), frame='icrs')

            # Transform the star's coordinates to AltAz for the current observing time
            coords = coordr.transform_to(aa)

            # Update the observing time for the next data point
            observing_time = timek[i + 1]
            aa = AltAz(location=observing_location, obstime=observing_time)

            # Transform the star's coordinates for the next time step
            coordr = SkyCoord(ra=val[0], dec=val[1], unit=(u.deg, u.deg), frame='icrs')
            coords2 = coordr.transform_to(aa)

            # Interpolate intermediate data points to describe the motion of the star over time
            temp_az, temp_alt, pltxt, pltyt = Make_Arc_and_Move(zk[i], tk[i], coords.az.deg, coords.alt.deg,
                                                                 zd[i], td[i], zd[i], coords2.az.deg,
                                                                 coords2.alt.deg, star_azs, star_alts, PixScale,
                                                                 DectX, DectY)
            tamp, tstarx, tstary, txstd, tystd = MakeStarTrailTable(val[2], temp_az, temp_alt, RefMag, RefCount,
                                                                     ExposureTime, std)
            # Store star data if valid
            if min(tstarx) > 0 and min(tstary) > 0:
                amps = np.append(amps, tamp)
                xstars = np.append(xstars, tstarx)
                ystars = np.append(ystars, tstary)
                xstds = np.append(xstds, txstd)
                ystds = np.append(ystds, tystd)
                star_alts = np.append(star_alts, temp_alt)
                star_azs = np.append(star_azs, temp_az)
                star_alts = np.append(star_alts, np.nan)
                star_azs = np.append(star_azs, np.nan)

            cenx, ceny, AltDiff, AzDiff = zk[i], tk[i], td[i], zd[i]
            SatPosition = SkyCoord(ra=cenx, dec=ceny, unit=(u.deg, u.deg))
        amps = np.array(amps) 
        xstds = np.array(xstds) 
        ystds = np.array(ystds) 
        xstars = np.array(xstars) 
        ystars = np.array(ystars) 
        shaper = [DectY, DectX]
        saver = [shaper, amps, xstars, ystars, xstds, ystds]    
        saver = np.array(saver, dtype = "object")
        # Save data to a file
        np.save("Table/" + str(i) + ".npy", saver, allow_pickle=True)
        
        LatestSatPosition = SkyCoord(ra=cenx + AzDiff, dec=ceny + AltDiff, unit=(u.deg, u.deg))
        XDrift, YDrift = Pix_Sep(SatPosition, LatestSatPosition, PixScale)
        print(XDrift, YDrift)
        print(AzDiff, AltDiff)
        print(i)
        print("_______________________________- ")



satellite, zk, tk, zd, td, rk, dk, timek = GetSats(stations_url, lat,long,elev,hrs,mins,secs, year, month, day)
GetStars_Plot(lat,long, elev,rk, dk, timek, zk, tk)


    
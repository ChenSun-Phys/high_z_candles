J/A+A/655/A109  Chandra view of the LX-LUV relation in quasars  (Bisogni+, 2021)
================================================================================
The Chandra view of the relation between X-ray and UV emission in quasars.
    Bisogni S., Lusso E., Civano F., Nardini E., Risaliti G., Elvis M.,
    Fabbiano G.
    <Astron. Astrophys. 655, A109 (2021)>
    =2021A&A...655A.109B        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: QSOs ; Active gal. nuclei ; Redshifts ; X-ray sources ;
              Ultraviolet ; Optical; Surveys; Spectroscopy; Photometry
Keywords: galaxies: active - galaxies: nuclei - quasars: general -
          quasars: supermassive black holes

Abstract:
    We present a study of the relation between X-rays and ultraviolet
    emission in quasars for a sample of broad-line, radio-quiet objects
    obtained from the cross-match of the Sloan Digital Sky Survey DR14
    with the latest Chandra Source Catalog 2.0 (2332 quasars) and the
    Chandra COSMOS Legacy survey (273 quasars). The non-linear relation
    between the ultraviolet (at 2500{AA}, LUV) and the X-ray (at 2keV, LX)
    emission in quasars has been proved to be characterised by a smaller
    intrinsic dispersion than the observed one, as long as a homogeneous
    selection, aimed at preventing the inclusion of contaminants in the
    sample, is fulfilled. By leveraging on the low background of Chandra,
    we performed a complete spectral analysis of all the data available
    for the SDSS-CSC2.0 quasar sample (i.e. 3430 X-ray observations), with
    the main goal of reducing the uncertainties on the source properties
    (e.g. flux, spectral slope). We analysed whether any evolution of the
    LX-LUV relation exists by dividing the sample in narrow redshift
    intervals across the redshift range spanned by our sample, z~=0.5-4.
    We find that the slope of the relation does not evolve with redshift
    and it is consistent with the literature value of 0.6 over the
    explored redshift range, implying that the mechanism underlying the
    coupling of the accretion disc and hot corona is the same at the
    different cosmic epochs. We also find that the dispersion decreases
    when examining the highest redshifts, where only pointed observations
    are available. These results further confirm that quasars are
    'standardisable candles', that is we can reliably measure cosmological
    distances at high redshifts where very few cosmological probes are
    available.

Description:
    Optical and X-ray properties of the 3430 X-ray observations,
    corresponding to 2332 SDSS DR14 sources, analysed. For each entry we
    list SDSS name, redshift, Chandra observation identifier, off-axis
    angle in the X-ray observation, exposure time in the X-ray
    observation, number of the SDSS Data Release, UV rest-frame
    monochromatic fluxes at 2500 Angstroem, X-ray rest-frame monochromatic
    fluxes at 2keV, photon index, rest-frame 2keV flux limit per
    observation, raw counts in the soft band (0.5-2keV), raw counts in
    the hard band (2-7keV), signal-to-noise in the soft band
    (0.5-2keV), signal-to-noise in the hard band (2-7keV).

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table2.dat       172     3430   Properties of the 3430 X-ray observations
--------------------------------------------------------------------------------

See also:
  VII/286 : SDSS quasar catalog, fourteenth data release (Paris+, 2018)

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units       Label        Explanations
--------------------------------------------------------------------------------
   1- 18  A18   ---         Name         Quasar name in DR14,
                                          HHMMSS.ss+DDMMSS.s
  20- 27  F8.6  ---         z            Redshift
  29- 33  I5    ---         ObsId        Chandra Observation Identifier
  35- 43  F9.7 [arcmin]     theta        Off-axis angle of source
                                          in X-ray observation
  45- 53  F9.2  [s]         ExpTime      Exposure time of X-ray obs.
  55- 58  A4    ---         DRflag       Number of SDSS Data Release,
                                          DR7, DR12 or DR14
  60- 67  F8.4 [mW/m2/Hz]   logF2500A    Rest-frame fluxes at 2500 Angstroem (1)
  69- 74  F6.4 [mW/m2/Hz] e_logF2500A    Error on rest-frame fluxes at 2500{AA}
  76- 83  F8.4 [mW/m2/Hz]   logF2keV     Rest-frame fluxes at 2keV
  85- 91  F7.4 [mW/m2/Hz] e_logF2keV     Lower error on fluxes at 2keV
  93- 98  F6.4 [mW/m2/Hz] E_logF2keV     Upper error on fluxes at 2keV
 100-107  F8.6  ---         Gamma        Photon index
 109-116  F8.5  ---       e_Gamma        Lower error on photon index
 118-125  F8.5  ---       E_Gamma        Upper error on photon index
 126-134  F9.4  ---         logF2keVlim  ? Rest-frame 2keV flux limit
                                          per observation
     136  A1    ---       n_logF2keVlim  [I] I for -Infinity
 137-144  F8.3  ---         Cts          Raw counts in soft band (0.5-2keV)
 146-153  F8.3  ---         Cth          Raw counts in hard band (2-7keV)
 155-162  F8.5  ---         SNs          SN in the soft band (0.5-2keV)
 164-172  F9.6  ---         SNh          ? SN in hard band (2-7keV)
--------------------------------------------------------------------------------
Note (1): Fluxes are in units of log(erg/s/cm^2^/Hz).
--------------------------------------------------------------------------------

Acknowledgements:
    Susanna Bisogni, susanna.bisogni(at)inaf.it

================================================================================
(End)                                        Patricia Vannier [CDS]  08-Sep-2021

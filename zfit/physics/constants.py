# Masses [GeV]
mMu    = Const( 0.105658 )
mE     = Const( 0.000511 )
MB     = Const( 5.27958 )
MKst   = Const( 0.89594 )
Mb     = Const( 4.16992 )
Ms     = Const( 0.0796146 )
MD     = Const( 1.8648 )

# for Breit-Wigner  [GeV]  Attention to the units!! (KpiMass is defined in MeV)
mPi     = Const( 0.1395706) # pion
mK      = Const( 0.49368)   # kaon
WKst    = Const( 0.0487) # K* width
dBmeson = Const(  5  )   # GeV^-1 (meson radius) << irrelevant since LBmeson = 0  =>  Blatt-Weisskopt barrier factor = (p*d)^0
dRes    = Const( 1.6 )   # GeV^-1 (meson radius)  
LBmeson = 0  # (Maximum) angular momentum
LKst    = 1  # (Maximum) angular momentum

# for LASS parametrization  [GeV]  Attention to the units!! (KpiMass is defined in MeV)
MKst1430  = Const( 1.425 )
WKst1430  = Const( 0.270 )
LKst1430  = 0
# Empirical parameters
aLASS  =  Const(3.83)  # GeV*c (values from LHCb-PAPER-2016-012)
rLASS  =  Const(2.86)  # GeV*c 

# from PDG
MJpsi  = Const(3.0969)
Mpsi2S = Const(3.6861)
WJpsi  = Const(92.9*1e-6) # 92.9 keV
Wpsi2S = Const(296*1e-6) # 296 keV
  
GF      = Const( 1.1663787e-05 )
alpha_e = Const( 0.00751879699248 )
h_bar = Const( 6.582119 * 1e-25 )  # GeV*s
tauB   = Const( 1.520 * 1e-12 ) # s

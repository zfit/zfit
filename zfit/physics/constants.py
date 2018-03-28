import tensorflow as tf

# Masses [GeV]
mMu    = tf.constant( 0.105658  , tf.float64 )
mE     = tf.constant( 0.000511  , tf.float64 )
MB     = tf.constant( 5.27958   , tf.float64 )
MKst   = tf.constant( 0.89594   , tf.float64 )
Mb     = tf.constant( 4.16992   , tf.float64 )
Ms     = tf.constant( 0.0796146 , tf.float64 )
MD     = tf.constant( 1.8648    , tf.float64 )

# for Breit-Wigner  [GeV]  Attention to the units!! (KpiMass is defined in MeV)
mPi     = tf.constant( 0.1395706 , tf.float64 ) # pion
mK      = tf.constant( 0.49368 , tf.float64 )   # kaon
WKst    = tf.constant( 0.0487 , tf.float64 ) # K* width
dBmeson = tf.constant(  5  , tf.float64 )   # GeV^-1 (meson radius) << irrelevant since LBmeson = 0  =>  Blatt-Weisskopt barrier factor = (p*d)^0
dRes    = tf.constant( 1.6 , tf.float64 )   # GeV^-1 (meson radius)  
LBmeson = 0  # (Maximum) angular momentum
LKst    = 1  # (Maximum) angular momentum

# for LASS parametrization  [GeV]  Attention to the units!! (KpiMass is defined in MeV)
MKst1430  = tf.constant( 1.425  , tf.float64 )
WKst1430  = tf.constant( 0.270  , tf.float64 )
LKst1430  = 0
# Empirical parameters
aLASS  =  tf.constant(3.83 , tf.float64 )  # GeV*c (values from LHCb-PAPER-2016-012)
rLASS  =  tf.constant(2.86 , tf.float64 )  # GeV*c 

# from PDG
MJpsi  = tf.constant(3.0969 , tf.float64 )
Mpsi2S = tf.constant(3.6861 , tf.float64 )
WJpsi  = tf.constant(92.9*1e-6 , tf.float64 ) # 92.9 keV
Wpsi2S = tf.constant(296*1e-6  , tf.float64 ) # 296 keV
  
GF      = tf.constant( 1.1663787e-05  , tf.float64 )
alpha_e = tf.constant( 0.00751879699248  , tf.float64 )
h_bar   = tf.constant( 6.582119 * 1e-25  , tf.float64 )  # GeV*s
tauB    = tf.constant( 1.520 * 1e-12  , tf.float64 ) # s

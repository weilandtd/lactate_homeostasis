import numpy as np 
from scipy.integrate import odeint

flux_names = ['vG0', 'vF0', 'vG', 'vL', 'vF']

def fluxes(x,p):
    G,L,F,I = x

    kG, kL, kF, kR, vG0, _vF0, vE, K_G_L, K_F0_L, K_F0_I, K_G_I, a, Imax, deltaI = p

   # Glucose production in liver
    vG0 = vG0

    # Lipolysis in adipose tissue
    vF0 = _vF0 * ( (1/(1+L/K_F0_L))*(1/(1+I/K_F0_I)) - kR * F)

    # Glycolysis
    vG = kG * G * (1/(1+L/K_G_L)) * ( 1 + a * I / (I + K_G_I))

    # Lactate consumption
    vL = vE * kL*L/(kL*L + kF*F)

    # NEFA consumption
    vF = vE * kF*F/(kL*L + kF*F)

    return [vG0, vF0, vG, vL, vF]


parameter_names = ['kG', 'kL', 'kF','kR', 'vG0', 'vF0', 'vE', 'KI_G_L', 'KI_F0_L', 'KI_F0_I','KA_G_I', 'a', 'Imax', 'deltaI']

def insulin(G, I0, deltaI):
    k = 4
    C = 2.3
    return I0 * abs(G)**k / (abs(G)**k + C**k) + deltaI

def reference_state(noise = 0.1 ):
    G0 = 1.0
    L0 = 1.0
    F0 = 1.0
    X0 = np.array([G0, L0, F0]) * (1 + noise * np.random.randn(3))
    I0 = insulin(X0[0], 1.0, 0.0)
    return np.concatenate([X0, [I0]])

def reference_fluxes(noise= 0.1):
    # Fluxes
    vG0 = 0.3 / 2.0
    vF0 = 0.7 * 3/16
    vG = vG0
    vL = vG0 * 2.0
    vF = 0.7
    
    return np.array([vG0, vF0, vG, vL, vF]) * (1 + noise * np.random.randn(5))


def parameter_dict(p, parameter_names=parameter_names):
    return {pn:pi for pn,pi in zip(parameter_names,p)}


def change_parameters(p,e=[1.0,],ix=["vE",], fold_change=False):
    p_c = p.copy()
    for this_e, this_ix in zip(e,ix):
        i = parameter_names.index(this_ix)
        if fold_change:
            p_c[i] = p_c[i] * this_e
        else:
            p_c[i] = this_e
    return p_c


def fluxes(x,p):
    G,L,F,I = x

    kG, kL, kF, kR, vG0, _vF0, vE, K_G_L, K_F0_L, K_F0_I, K_G_I, a, Imax, deltaI = p

   # Glucose production in liver
    vG0 = vG0

    # Lipolysis in adipose tissue
    vF0 = _vF0 * ( (1/(1+L/K_F0_L))*(1/(1+I/K_F0_I)) - kR * F)

    # Glycolysis
    vG = kG * G * (1/(1+L/K_G_L)) * ( 1 + a * I / (I + K_G_I))

    # Lactate consumption
    vL = vE * kL*L/(kL*L + kF*F)

    # NEFA consumption
    vF = vE * kF*F/(kL*L + kF*F)

    return [vG0, vF0, vG, vL, vF]
    

#############################
# Time scales for the model #
#############################

# We match the time scale based on turn over rates
# Lactate:
# T = X0/V0 = 1.0/0.3[T] -> T = 3.3 [T] 
# To scale from [T] to minutes won consider 
# Lactate turnover == T
# Thus scaling factor of from simulation time to minutes 
# Lactate
# tauL = T_lac / T  = T_lac /X0 * V0 = 5.0 (min) * 0.3 [1/T] -> 1.5 (min)/[T]   
# Glucose
# tauG = 14.0 (min) * 0.15 [1/T] -> 2.1 (min)/[T]
# FFA
# tauF = 10.0 (min) * 0.7 [1/T] -> 7.0 (min)/[T]
# Insulin
# tauI = 1.0 (min) -> 1.0 (min)/[T]

# From volume of distribution mesrurements
tauG = 14.0  * 0.15
tauL = 5.0   * 0.3
tauF = 10.0  * 0.7*3/16/2*3
tauI = 1.0 # Insulin is fast

# Metformin
tauM = 120 # Metformin is slow

##############################
# Differential equations     #
##############################
def equation(x,t,p):
    G,L,F,I = x
    
    vG0, vF0, vG, vL, vF = fluxes(x,p)

    dGdt = vG0- vG
    dLdt = 2*vG - vL
    dFdt = vF0 - vF * 3/16
    
    # Dynamics of insulin
    Imax = p[parameter_names.index('Imax')]
    dI = p[parameter_names.index('deltaI')]
    dIdt = insulin(G,Imax,dI) - I

    return [dGdt/tauG, dLdt/tauL, dFdt/tauF, dIdt/tauI] # Scale the time scales to minutes
    

def equation_clamped(x,t,p):
    G,L,F,I = x
    
    vG0, vF0, vG, vL, vF = fluxes(x,p)

    dGdt = vG0 - vG
    dLdt = 2*vG  - vL
    dFdt = vF0 - vF * 3/16
    
    # Dynamics of insulin
    Imax = p[parameter_names.index('Imax')]
    dI = p[parameter_names.index('deltaI')]
    dIdt = insulin(G,Imax,dI) - I

    dGdt = np.clip(dGdt,0,np.inf)

    return [dGdt/tauG , dLdt/tauL, dFdt/tauF, dIdt/tauI] # Scale the time scales to minutes

def equation_metformin(x,t,p,metformin_in):
    G,L,F,I,M = x
    
    vG0, vF0, vG, vL, vF = fluxes_metformin(x,p)
    

    dGdt = vG0- vG
    dLdt = 2*vG - vL
    dFdt = vF0 - vF * 3/16 
    
    # Dynamics of insulin
    Imax = p[parameter_names.index('Imax')]
    dI = p[parameter_names.index('deltaI')]
    dIdt = insulin(G,Imax,dI) - I
 
    dMdt = metformin_in - M

    return [dGdt/tauG, dLdt/tauL, dFdt/tauF, dIdt/tauI, dMdt/tauM] # Scale the time scales to minutes


def steady_state(p, t_max=500, concentration_noise=0.1):
    x0 = reference_state(concentration_noise)
    result = odeint(equation,x0, [0,t_max], args=(p,))[-1]
    return result


def response(p, t_max=500, n_data=1000, concentration_noise=0.1 , x0=None, type=None, metformin_in=0):

    if x0 is None:
        x0 = reference_state(concentration_noise)
        # Not nice but works
        if type == 'metformin':
            x0 = np.concatenate([x0, [0.0]])

    t = np.linspace(0,t_max,n_data)
    if type == 'clamped':
        result = odeint(equation_clamped,x0, t, args=(p,))
    elif type == 'metformin':
        result = odeint(equation_metformin,x0, t, args=(p,metformin_in))
    else:
        result = odeint(equation,x0, t, args=(p,))

    return t, result
    

def insulin_pulse(p, t_max=50, n_data=100, 
                  concentration_noise=0.1, 
                  pulse_time=10,
                  pulse_amplitude=0.2,
                  ):
    x0 = reference_state(concentration_noise)

    # Prepulse simulation
    t1 = np.linspace(0,pulse_time,n_data)
    res1 = odeint(equation,x0, t1, args=(p,))
    # Pulse simulation
    x0 = res1[-1]
    x0[3] += pulse_amplitude # Add insulin pulse
    t2 = np.linspace(pulse_time,t_max,n_data)
    res2 = odeint(equation,x0, t2, args=(p,))
    # Concatenate
    t = np.concatenate((t1,t2))
    result = np.concatenate((res1,res2))

    return t, result


##############################
# Parameter constraints      #
##############################

# Calcualte parameters based on the steady state
def reference_parameters( concentration_noise = 0.1, flux_noise = 0.1, ki_noise = 0.1):
    # Reference state
    X0 = reference_state(concentration_noise)
    G0, L0, F0, I0 = X0
    #Ref inuslin 
    Imax = 1.0
    deltaI = 0.0

    # Reference fluxes
    vG0, vF0, vG, vL, vF = reference_fluxes(flux_noise)

    # Parameters 
    vE = 1.0
    noise = np.random.lognormal(0,ki_noise,5) 

    K_G_L = L0 * noise[0] * 2.0
    K_F0_L = L0 * noise[1] * 5.0
    K_F0_I = insulin(G0, Imax, deltaI)* noise[2] 
    K_G_I = insulin(G0, Imax, deltaI) * noise[3] * 20.0
    kG, kL, kF = 1.0, 1.0, 1.0

    a = 10 

    # We assume 2/3 of the lipolysis is re-esterified
    kR = (1/(1+L0/K_F0_L)) * (1/(1+insulin(G0,Imax,deltaI)/K_F0_I)) * 2/3

    p = [kG, kL, kF, kR, 1.0, 1.0, vE, K_G_L, K_F0_L, K_F0_I, K_G_I, a, Imax, deltaI]

    _vG0, _vF0, _vG, _vL, _vF = fluxes(X0, p)
    
    # Flux closure
    vG0 = vG0/_vG0
    vF0 = vF0/_vF0
    kG = vG/_vG
    kL = vL/_vL
    kF = vF/_vF

    return [kG, kL, kF, kR, vG0, vF0, vE, K_G_L, K_F0_L, K_F0_I, K_G_I, a, Imax, deltaI]



def fluxes_metformin(x,p):
    G,L,F,I,M = x

    kG, kL, kF, kR, vG0, _vF0, vE, K_G_L, K_F0_L, K_F0_I, K_G_I, a, Imax, deltaI = p

   # Glucose production in liver
    vG0 = vG0

    # Lipolysis in adipose tissue
    vF0 = _vF0 * ( (1/(1+L/K_F0_L))*(1/(1+I/K_F0_I)) - kR * F)

    # Glycolysis
    vG = kG * G * (1/(1+L/K_G_L)) * ( 1 + a * I / (I + K_G_I))

    # Lactate consumption
    vL = vE * kL*L/(kL*L + kF*F) * 1 / (1 + M)

    # NEFA consumption
    vF = vE * kF*F/(kL*L + kF*F) * 1 / (1 + M)

    return [vG0, vF0, vG, vL, vF]
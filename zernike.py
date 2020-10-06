import numpy as np
from math import factorial as fact

def zernike_pol(rho, theta, n, m):
    """Evaluate the normalized radial zernike polynomials in the unit 
    circle using the radial and azimuthal index convention outlined in 
    https://en.wikipedia.org/wiki/Zernike_polynomials. 

    Arguments:
    rho -- radial mesh covering the unit range [0, 1]
    theta -- polar angle mesh covering the range [0, 2*pi]
    n, m -- integer radial Zernike polynomial indices
    """
    def R(r, i, j):
        radial = 0
        for s in range((i - np.abs(j))//2 + 1):
            num = (-1)**s * fact(i - s) * r**(i - 2*s)
            den = fact(s) * fact((i + np.abs(j) - 2*s)//2) * fact((i - np.abs(j) - 2*s)//2)
            radial += num / den
        return radial

    def Norm(i, j):        
        if j == 0:
            return np.sqrt(2*(i + 1) / (1 + 1))
        else:
            return np.sqrt(2*(i + 1))

    amplitude = Norm(n, m) * R(rho, n, m) 
    if m < 0:
        return amplitude * np.sin(m*theta)
    else:
        return amplitude * np.cos(m*theta)
    
def zernike_expansion(rho, theta, coeffs):
    """Compute zernike expansion for a set of coefficients over the 
    pupil coords. An example for the coeffs argument is:  
    
        coeffs = {'Z00':0.1, 'Z20':0.012, 'Z1m1':-0.15} 
    
    where the keys must be strings in the 'Znm' format denoting the 
    (n, m)-th term, and the corresponding value. For negative m-indices
    use the 'Zxmx' format, with an 'm' between (n, m) for 'minus'.

    Arguments:
    rho -- radial mesh covering the unit range [0, 1]
    theta -- polar angle mesh covering the range [0, 2*pi]
    coeffs -- dictionary containing the Zernike coefficients
    """
    expansion = 0.
    for coeff_key, coeff_val in coeffs.items():
        n, m = int(coeff_key.strip('Z')[0]), int(coeff_key.strip('Z')[-1])
        if 'm' in coeff_key:
            m = -m
        expansion += coeff_val * zernike_pol(rho, theta, n, m)
    return expansion

if __name__ in '__main__':
    import matplotlib.pyplot as plt

    r, t = np.linspace(0, 1, 2**8), np.linspace(-np.pi, np.pi, 2**8)
    R, T = np.meshgrid(r, t)

    # Two lists of coefficients up to some order

    fourth_order_coeffs = {
                                                     'Z00':1.0,
                                                'Z1m1':0.0, 'Z11':0.0,
                                        'Z2m2':3.0, 'Z20':1.0, 'Z22':-1.5,
                                    'Z3m3':0.1, 'Z3m1':1.0, 'Z31':0.5, 'Z33':0.3,
                             'Z4m4':1.5, 'Z4m2':3.0, 'Z40':2.00, 'Z42':1.0, 'Z44':0.2,
    }

    eigth_order_coeffs = {
                                                     'Z00':1.0,
                                                'Z1m1':0.0, 'Z11':0.0,
                                        'Z2m2':3.0, 'Z20':1.0, 'Z22':-1.5,
                                    'Z3m3':0.1, 'Z3m1':1.0, 'Z31':0.5, 'Z33':0.3,
                             'Z4m4':1.5, 'Z4m2':3.0, 'Z40':2.00, 'Z42':1.0, 'Z44':0.2,
                        'Z5m5':0.0, 'Z5m3':0.0, 'Z5m1':0.0, 'Z51':0.0, 'Z53':0.0, 'Z55':0.0,
                 'Z6m6':-0.5, 'Z6m4':0.0, 'Z6m2':0.0, 'Z60':0.0, 'Z62':0.0, 'Z64':0.0, 'Z66':0.0,
            'Z7m7':0.0, 'Z7m5':0.0, 'Z7m3':2.0, 'Z7m1':0.0, 'Z71':0.0, 'Z73':0.0, 'Z75':0.0, 'Z77':0.0,
     'Z8m8':0.0, 'Z8m6':0.0, 'Z8m4':0.0, 'Z8m2':0.0, 'Z80':-1.0, 'Z82':0.0, 'Z84':0.0, 'Z86':0.0, 'Z88':0.0,
    }

    fourth_order = zernike_expansion(R, T, fourth_order_coeffs)
    eigth_order = zernike_expansion(R, T, eigth_order_coeffs)

    plt.figure()
    ax = plt.subplot(121, polar=True)
    ax.set_title('Fourth-order')
    ax.pcolor(T, R, fourth_order, cmap='RdBu', shading='auto')
    ax.set_yticks([])
    ax.set_xticks([])
    ax = plt.subplot(122, polar=True)
    ax.set_title('Eigth-order')
    ax.pcolor(T, R, eigth_order, cmap='RdBu', shading='auto')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()
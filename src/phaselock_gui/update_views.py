import pyqtgraph as pg
import numpy as np
import py4DSTEM
from skimage.color import lab2rgb
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import warnings

from phaselock_gui.utils import pg_point_roi

def complex_to_Lab(im,amin=None,amax=None,gamma=1,L_scale=100,ab_scale=64,uniform_L=None):
    Lab = np.zeros(im.shape + (3,),dtype=np.float64)
    angle = np.angle(im)
    
    L = Normalize(vmin=amin, vmax=amax, clip=True)(np.abs(im)) ** gamma
    L = Normalize()(L)
    
    # attempt at polynomial saturation
    ab_prescale = 4*L - 4*L*L
    
    Lab[...,0] = uniform_L or L * L_scale
    Lab[...,1] = np.cos(angle) * ab_scale * ab_prescale
    Lab[...,2] = np.sin(angle) * ab_scale * ab_prescale
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rgb = lab2rgb(Lab)
    
    return rgb

def update_real_space_view(self, reset=False):
    scaling_mode = self.vimg_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"], scaling_mode

    snapping_mode = self.snapping_group.checkedAction().text().replace("&", "")
    assert snapping_mode in ["None", "Center of Mass", "Maximum"], snapping_mode

    if self.image is None:
        return

    # get the object and its FFT
    obj = self.image
    objfft = np.fft.fft2(obj)

    # generate corrdinates
    qx,qy = np.meshgrid(np.fft.fftfreq(obj.shape[0]),np.fft.fftfreq(obj.shape[1]),indexing='ij')
    qr = np.hypot(qx,qy)
    rx,ry = np.meshgrid(np.arange(obj.shape[0]),np.arange(obj.shape[1]),indexing='ij')
    rr = np.hypot(rx,ry)

    # get coordinates and radius from the ROI selector
    (slice_x, slice_y), _ = self.virtual_detector_roi.getArraySlice(
        self.image, self.diffraction_space_widget.getImageItem()
    )
    x0 = (slice_x.start + slice_x.stop) / 2.0
    y0 = (slice_y.start + slice_y.stop) / 2.0
    R = (slice_y.stop - slice_y.start) / 2.0

    self.diffraction_space_view_text.setText(f"[({x0},{y0}),{R}]")

    # convert ROI pixel coordinates to Fourier units
    cx = np.fft.fftshift(qx)[int(x0),int(y0)]
    cy = np.fft.fftshift(qy)[int(x0),int(y0)]
    sigma = R / obj.shape[0] / 2 # Extra factor of 2 to make the mask roughly fit in the GUI circle ROI
    # print(f"cx:{cx}, cy:{cy}, sigma:{sigma}")

    dq = np.hypot(qx - cx, qy - cy)
    mask_approx = dq < sigma

    # refine the center coordinate
    if snapping_mode == "None":
        cx_refine, cy_refine = cx, cy
    elif snapping_mode == "Center of Mass":
        cxrp, cyrp = py4DSTEM.process.utils.get_CoM(np.fft.fftshift(np.abs(objfft) * mask_approx))
        # convert back to un-fftshifted coordinates
        cxrp -= obj.shape[0]//2 # does this work right for even/odd sizes?
        cyrp -= obj.shape[1]//2
        cx_refine, cy_refine = qx[int(cxrp),int(cyrp)], qy[int(cxrp),int(cyrp)]
    elif snapping_mode == "Maximum":
        # refine the center coordinate using the max of the masked area
        test_mask = gaussian_filter(np.abs(objfft) * mask_approx,2)
        cxrp, cyrp = np.unravel_index(np.argmax(test_mask,axis=None),objfft.shape)
        cx_refine, cy_refine = qx[int(cxrp),int(cyrp)], qy[int(cxrp),int(cyrp)]
    else:
        raise ValueError(f"Uh-oh... we should not be here! Snapping mode {snapping_mode}")

    dq = np.hypot(qx - cx_refine, qy - cy_refine)

    # update ROI selector after snapping
    x0_snap = np.argmin(np.abs(cx - np.fft.fftshift(qx)[:,0])) - R
    y0_snap = np.argmin(np.abs(cy - np.fft.fftshift(qy)[0,:])) - R
    # print(f"Snap pixels: {x0_snap,y0_snap}")
    self.virtual_detector_roi.setPos(y0_snap,x0_snap,finish=False)

    mask = np.exp(-0.5 * dq**2 / sigma**2)/(sigma * np.sqrt(2.0 * np.pi))
    mask_max = np.argmax(mask)

    dqi = np.hypot(qx + cx, qy + cy)
    maski = np.exp(-0.5 * dqi**2 / sigma**2)/(sigma * np.sqrt(2.0 * np.pi))
    maski_max = np.argmax(maski)

    mask_pair = np.minimum(mask + maski, 1)

    A = np.fft.ifft2(objfft * mask_pair).real
    X = A * np.cos(2.0*np.pi * (cx*rx + cy*ry))
    Y = A * np.sin(2.0*np.pi * (cx*rx + cy*ry))

    lp_w = 1.0
    lowpass_mask = qr <= np.hypot(lp_w*cx,lp_w*cy)
    Xf = np.fft.ifft2(np.fft.fft2(X)*lowpass_mask).real
    Yf = np.fft.ifft2(np.fft.fft2(Y)*lowpass_mask).real

    phi = np.arctan2(-Yf,Xf)
    
    # generate phase overlay
    phase_image = complex_to_Lab(np.exp(1j*phi),uniform_L=60)


    if scaling_mode == "Linear":
        new_view = obj
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(obj, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(obj, 0))
    else:
        raise ValueError("Mode not recognized")

    blended_image = phase_image
    self.real_space_widget.setImage(
        blended_image,
        autoLevels=True,
    )

    # Show the mask for coordinate debugging
    unmaskedFFT = np.abs(np.fft.fftshift(np.fft.fft2(self.image)))
    maskedFFT = unmaskedFFT*np.fft.fftshift(mask_pair)
    new_view = 2*maskedFFT + unmaskedFFT
    levels = (np.min(new_view), np.percentile(new_view,99.9))
    self.diffraction_space_widget.setImage(
        new_view, levels=levels, autoRange=False,
    )


def update_diffraction_space_view(self, reset=False):
    scaling_mode = self.diff_scaling_group.checkedAction().text().replace("&", "")
    assert scaling_mode in ["Linear", "Log", "Square Root"]

    if self.image is None:
        return

    DP = np.abs(np.fft.fftshift(np.fft.fft2(self.image)))

    if scaling_mode == "Linear":
        new_view = DP
    elif scaling_mode == "Log":
        new_view = np.log2(np.maximum(DP, self.LOG_SCALE_MIN_VALUE))
    elif scaling_mode == "Square Root":
        new_view = np.sqrt(np.maximum(DP, 0))
    else:
        raise ValueError("Mode not recognized")

    levels = (np.min(new_view), np.percentile(new_view,99.9))
    print(f"Levels: {levels}")

    self.diffraction_space_widget.setImage(
        new_view, autoLevels=not levels, levels=levels, autoRange=reset
    )



# https://github.com/basler/pypylon/blob/master/samples/guiimagewindow.py
# https://github.com/basler/pypylon/blob/master/samples/opencv.py
# https://snyk.io/advisor/python/pypylon/functions/pypylon.pylon.TlFactory

import numpy as np, time, cv2, os

from pypylon import pylon, genicam
from datetime import datetime
from PIL import Image
from argparse import ArgumentParser#, Namespace

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()

DATETIME = datetime.now().strftime("%d-%m-%Y %H-%M-%S")  # dd/mm/YY H:M:S
BASE_PATH = os.getcwd()
SAVE_PATH = BASE_PATH + '/images/' + DATETIME
NUM_CAMERAS = 2

def get_parser() -> ArgumentParser:
    """
    Examples:
        parser.add_argument("-fr", "--frame_rate", type=int, choices=[0, 1, 2], help="...helpful statement here...")
    """
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=False, help="Record and save video instead of individual images? [DEFAULT: False]")
    parser.add_argument("--frame_rate", type=float, default=1.0, help="Frame rate (fps) for image capture? [DEFAULT: 1.0 fps]")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")
    parser.add_argument("--fov", type=int, default=2000, help="Field of view (mm) for single camera? [DEFAULT: 2000 mm]")
    parser.add_argument("--speed", type=int, default=15, help="Object speed (m/min)? [DEFAULT: 15 m/min]")
    parser.add_argument("--max_time", type=int, default=60, help="Maximum time for single batch (mins)? [DEFAULT: 60 mins]")
    parser.add_argument("--brightness", type=float, default=0.3, help="Threshold for AutoTargetBrightness pylon function (average gray level of pixels)? [DEFAULT: 0.3]")
    parser.add_argument("--cam_setup", type=bool, default=False, help="Setup camera parameters? [DEFAULT: False]")

    return parser

class BaslerCam: #object
    def __init__(self, args):
        self.video = args.video
        self.frame_rate = args.frame_rate
        self.fov = args.fov
        self.speed = (args.speed / 60) * 1000  # converting m/min to mm/s
        self.max_time = args.max_time
        self.brightness = args.brightness

        factory = pylon.TlFactory.GetInstance()
        available_cameras = factory.EnumerateDevices()
        assert len(available_cameras) == NUM_CAMERAS, "Check camera connections..."

        self.cameras = {}
        for idx in range(len(available_cameras)):
            self.cameras[idx] = pylon.InstantCamera(factory.CreateDevice(available_cameras[idx]))
            # self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.cameras[idx].Open()
            print("Using device ", self.cameras[idx].GetDeviceInfo().GetModelName())  # Print the model name of the camera.

            self.max_expose = self.calc_exposure(idx)
            if args.cam_setup: self.camera_setup(idx)

    def calc_exposure(self, idx):
        w,h = self.cameras[idx].Width.GetValue(), self.cameras[idx].Height.GetValue()
        pixel = self.fov / w  # mm/px
        # fov_vert = pixel * h  # mm
        max_expose = (1 / (self.speed / pixel)) * 1000  # ms/px
        return max_expose

    def camera_setup(self, idx):
        minLowerLimit = self.cameras[idx].AutoGainLowerLimit.GetMin()
        maxUpperLimit = self.cameras[idx].AutoGainUpperLimit.GetMax()
        self.cameras[idx].AutoGainLowerLimit.SetValue(minLowerLimit)
        self.cameras[idx].AutoGainUpperLimit.SetValue(maxUpperLimit)

        minLowerLimit = self.cameras[idx].AutoExposureTimeLowerLimit.GetMin()
        self.cameras[idx].AutoExposureTimeLowerLimit.SetValue(minLowerLimit)
        self.cameras[idx].AutoExposureTimeUpperLimit.SetValue(self.max_expose*1000)

        self.cameras[idx].AutoTargetBrightness.SetValue(self.brightness)
        # camera.AutoFunctionROISelector.SetValue(AutoFunctionROISelector_ROI1)
        # camera.AutoFunctionROIUseBrightness.SetValue(True)
        self.cameras[idx].BalanceWhiteAuto.SetValue("Continuous")
        self.cameras[idx].GainAuto.SetValue("Continuous")  # SetValue("Once")
        self.cameras[idx].ExposureAuto.SetValue("Continuous")
        print("Exposure time is {} and max_exposure is {}".format((self.cameras[idx].ExposureTime.GetValue() / 1000), self.max_expose))
        # assert (self.cameras[idx].ExposureTime.GetValue() / 1000) < self.max_expose, "Exposure time needs to be smaller to avoid motion blur."

        self.cameras[idx].AcquisitionFrameRateEnable.SetValue(True)
        self.cameras[idx].AcquisitionFrameRate.SetValue(self.frame_rate)
        print("Frame rate set to: {} fps".format(self.cameras[idx].AcquisitionFrameRate.GetValue()))

    def get_display_image(self, imgs):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        colour = (0, 255, 0)
        thickness = 12
        crop = 0.8
        w,h = self.cameras[0].Width.GetValue(), self.cameras[0].Height.GetValue()
        disp_img = np.zeros((h, int(crop*w*len(imgs)), 3))
        start_pt = 0
        # print(np.array(imgs[0]).shape, disp_img.shape)
        # print(np.min(np.array(imgs[0])), np.max(np.array(imgs[0])))
        for i in range(len(imgs)):
            disp_img[:,start_pt:int(start_pt+w*crop),:] = np.array(imgs[i])[:,int(w*0.1):int(w*0.9),:]
            start_pt += int(w*crop)
        disp_img = np.array(disp_img).astype(np.uint8) #cv2.cvtColor(..., cv2.COLOR_BGR2RGB)
        disp_img = cv2.putText(disp_img, 'Press ESC to stop', (50, 150), font, font_scale, colour, thickness, cv2.LINE_AA)
        cv2.namedWindow('WINDOW', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("WINDOW", 1000, 600) # w, h
        cv2.imshow('WINDOW', disp_img)

    def grab_images(self):
        # converting to opencv bgr format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        for idx in range(len(self.cameras)):
            self.cameras[idx].StartGrabbing(pylon.GrabStrategy_OneByOne)
        start_time = time.time()
        timeout = time.time() + (self.max_time * 60)
        i = save_count = 0
        while (time.time() < timeout): # and self.camera.IsGrabbing():
            imgs = []
            for idx in range(len(self.cameras)):
                # Wait for an image and then retrieve it. A timeout of 5000ms is used --> needs to be greater than exposure time
                grab_result = self.cameras[idx].RetrieveResult(5000, pylon.TimeoutHandling_Return) #pylon.TimeoutHandling_ThrowException
                if grab_result.GrabSucceeded(): # Image grabbed successfully?
                    i += 1
                    img = converter.Convert(grab_result)
                    imgs.append(Image.fromarray(np.uint8(img.GetArray())))
                    if not (i % 20) or not ((i-1) % 20):
                        save_count += 1
                        print("Acquired {} frames in {:.0f} seconds \tSaving {}th image".format(i, time.time()-start_time, save_count))
                        file_name = str(idx) + '_' + ('000000' + str(save_count))[-6:] + '.png'
                        cv2.imwrite(file_name, np.array(imgs[idx]))

                else:
                    print("Couldn't grab, Error: ", grab_result.ErrorCode)  # grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError

                grab_result.Release()
                # time.sleep(0.05)

            self.get_display_image(imgs)

            k = cv2.waitKey(1)
            if k == 27:
                break

        print("Closing cameras...")
        for idx in range(len(self.cameras)):
            self.cameras[idx].Close()

        if time.time() > timeout: print("Timeout exceeded max batch time of {} minutes".format(self.max_time))

        print("FINISHED: Acquired {} frames in {:.0f} seconds".format(i, time.time()-start_time))

if __name__ == '__main__':

    os.mkdir(os.path.join(BASE_PATH, 'images', DATETIME))
    os.chdir(SAVE_PATH)

    args = get_parser().parse_args()

    bc = BaslerCam(args)
    bc.grab_images()

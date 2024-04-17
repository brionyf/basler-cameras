# https://github.com/basler/pypylon/blob/master/samples/guiimagewindow.py
# https://github.com/basler/pypylon/blob/master/samples/opencv.py

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

def get_parser() -> ArgumentParser:
    """
    Examples:
        parser.add_argument("-fr", "--frame_rate", type=int, choices=[0, 1, 2], help="...helpful statement here...")
    """
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default=False, help="Record and save video instead of individual images? [DEFAULT: False]")
    parser.add_argument("--frame_rate", type=float, default=0.5, help="Frame rate (fps) for image capture? [DEFAULT: 0.5 fps]")
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

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        print("Using device ", self.camera.GetDeviceInfo().GetModelName())  # Print the model name of the camera.

        self.max_expose = self.calc_exposure()
        if args.cam_setup: self.camera_setup()

    def calc_exposure(self):
        w,h = self.camera.Width.GetValue(), self.camera.Height.GetValue()
        pixel = self.fov / w  # mm/px
        # fov_vert = pixel * h  # mm
        max_expose = (1 / (self.speed / pixel)) * 1000  # ms/px
        return max_expose

    def camera_setup(self):
        minLowerLimit = self.camera.AutoGainLowerLimit.GetMin()
        maxUpperLimit = self.camera.AutoGainUpperLimit.GetMax()
        self.camera.AutoGainLowerLimit.SetValue(minLowerLimit)
        self.camera.AutoGainUpperLimit.SetValue(maxUpperLimit)

        minLowerLimit = self.camera.AutoExposureTimeLowerLimit.GetMin()
        self.camera.AutoExposureTimeLowerLimit.SetValue(minLowerLimit)
        self.camera.AutoExposureTimeUpperLimit.SetValue(self.max_expose*1000)

        self.camera.AutoTargetBrightness.SetValue(self.brightness)
        # camera.AutoFunctionROISelector.SetValue(AutoFunctionROISelector_ROI1)
        # camera.AutoFunctionROIUseBrightness.SetValue(True)
        self.camera.BalanceWhiteAuto.SetValue("Continuous")
        self.camera.GainAuto.SetValue("Continuous")  # SetValue("Once")
        self.camera.ExposureAuto.SetValue("Continuous")
        print("Exposure time is {} and max_exposure is {}".format((self.camera.ExposureTime.GetValue() / 1000), self.max_expose))
        assert (self.camera.ExposureTime.GetValue() / 1000) < self.max_expose, "Exposure time needs to be smaller to avoid motion blur."

        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
        print("Frame rate set to: {} fps".format(self.camera.AcquisitionFrameRate.GetValue()))

    def grab_images(self):
        # converting to opencv bgr format
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Image viewer variables
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 255, 0)
        thickness = 2

        # camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        start_time = time.time()
        i = 0
        timeout = time.time() + (self.max_time * 60)
        save_count = 0
        while (time.time() < timeout) and self.camera.IsGrabbing():

            # Wait for an image and then retrieve it. A timeout of 5000ms is used --> needs to be greater than exposure time
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_Return) #pylon.TimeoutHandling_ThrowException

            # Image grabbed successfully?
            if grab_result.GrabSucceeded():
                i += 1
                img = converter.Convert(grab_result)
                img = Image.fromarray(np.uint8(img.GetArray()))

                if not i % 10:
                    save_count += 1
                    print("Acquired {} frames in {:.0f} seconds \tSaving {}th image".format(i, time.time()-start_time, save_count))
                    file_name = ('000000' + str(save_count))[-6:] + '.png'
                    cv2.imwrite(file_name, np.array(img))

                img = cv2.putText(np.array(img), 'Press ESC to stop', (50, 50), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.namedWindow('WINDOW', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("WINDOW", 1000, 1000) # w, h
                cv2.imshow('WINDOW', img)

                k = cv2.waitKey(1)
                if k == 27:
                    break

            else:
                print("Couldn't grab, Error: ", grab_result.ErrorCode)  # grabResult.ErrorDescription does not work properly in python could throw UnicodeDecodeError

            grab_result.Release()
            # time.sleep(0.05)

        self.camera.Close()
        if time.time() > timeout: print("Timeout exceeded max batch time of {} minutes".format(self.max_time))
        print("FINISHED: Acquired {} frames in {:.0f} seconds".format(i, time.time()-start_time))

if __name__ == '__main__':

    os.mkdir(os.path.join(BASE_PATH, 'images', DATETIME))
    os.chdir(SAVE_PATH)

    args = get_parser().parse_args()

    bc = BaslerCam(args)
    bc.grab_images()

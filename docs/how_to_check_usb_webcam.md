* odroid에 연결되 usb webcam의 device number 확인 방법
  * "Found UVC 1.00 device SPC-A1200MB Webcam"와 같이 출력
    ```
    root@localhost:~# dmesg | grep video
    [    0.539664] Linux video capture interface: v2.00
    [    1.823830] exynos-drm-gsc 13e00000.video-scaler: drm gsc registered successfully.
    [    1.830228] exynos-drm-gsc 13e10000.video-scaler: drm gsc registered successfully.
    [    1.857274] exynos-drm-gsc 13e00000.video-scaler: The exynos gscaler has been probed successfully
    [    1.866003] exynos-drm exynos-drm: bound 13e00000.video-scaler (ops gsc_component_ops)
    [    1.873988] exynos-drm-gsc 13e10000.video-scaler: The exynos gscaler has been probed successfully
    [    1.882728] exynos-drm exynos-drm: bound 13e10000.video-scaler (ops gsc_component_ops)
    [    3.521499] s5p-jpeg 11f50000.jpeg: encoder device registered as /dev/video0
    [    3.527568] s5p-jpeg 11f50000.jpeg: decoder device registered as /dev/video1
    [    3.540264] s5p-jpeg 11f60000.jpeg: encoder device registered as /dev/video2
    [    3.547049] s5p-jpeg 11f60000.jpeg: decoder device registered as /dev/video3
    [    3.561953] s5p-mfc 11000000.codec: decoder registered as /dev/video4
    [    3.567397] s5p-mfc 11000000.codec: encoder registered as /dev/video5
    [    3.586414] usbcore: registered new interface driver uvcvideo
    [    4.561622] uvcvideo: Found UVC 1.00 device SPC-A1200MB Webcam (0c45:6340)
    ```
  * video# 확인
    ```
    root@localhost:~# ls -al /dev/v4l/by-id/
    total 0
    drwxr-xr-x 2 root root 60 Jan  1  2000 .
    drwxr-xr-x 4 root root 80 Jan  1  2000 ..
    lrwxrwxrwx 1 root root 12 Jan  1  2000 usb-Sonix_Technology_Co.__Ltd._SPC-A1200MB_Webcam-video-index0 -> ../../video6
    ```

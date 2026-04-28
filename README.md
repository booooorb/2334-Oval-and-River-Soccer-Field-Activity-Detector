Data source: https://www.richmond.ca/services/transportation/trafficcameras/Default.aspx

## Scheduled camera capture

The workflow in `.github/workflows/fetch-camera-image.yml` fetches the
Oval and River westbound traffic camera image every 5 minutes from 8:00 AM
to 10:00 PM PST.

```text
https://www.richmond.ca/trafficcam/vdd_oval_river_wb.jpg
```

Each captured JPG is committed to the `data` branch under `data/YYYY-MM-DD/`,
keeping camera-image commits out of `main`.

## Detector input preprocessing

Raw frames are reduced before they reach the detector:

```text
1280x720 full image
  -> top-left 540x200 crop
  -> black out irrelevant pixels with three polygons
```

## Simple local data flow

Run one command to fetch new raw images from GitHub's `data` branch, regenerate
processed images, and add any new rows to `labels/labels.csv`:

```powershell
& "D:\Downloads\a5_code_data\python-embed\python.exe" src\update_data.py
```

This creates:

```text
data/
  raw/YYYY-MM-DD/
  roi/YYYY-MM-DD/
  masked/YYYY-MM-DD/
labels/
  labels.csv
```

Edit only the `label` column in `labels/labels.csv` with `active`, `inactive`,
or `discard`. Syncing new raw data adds new label rows without removing existing
labels.


Data source: https://www.richmond.ca/services/transportation/trafficcameras/Default.aspx

## Scheduled camera capture

The workflow in `.github/workflows/fetch-camera-image.yml` fetches the
Oval and River westbound traffic camera image every 5 minutes, offset one
minute from the top of the hour:

```text
https://www.richmond.ca/trafficcam/vdd_oval_river_wb.jpg
```

Each captured JPG is uploaded as a GitHub Actions artifact and retained for
30 days. You can also run the workflow manually from the Actions tab with
the "Fetch camera image" workflow.

Data source: https://www.richmond.ca/services/transportation/trafficcameras/Default.aspx

## Scheduled camera capture

The workflow in `.github/workflows/fetch-camera-image.yml` fetches the
Oval and River westbound traffic camera image every 5 minutes from 8:00 AM
to 10:00 PM Vancouver time:

```text
https://www.richmond.ca/trafficcam/vdd_oval_river_wb.jpg
```

Each captured JPG is committed into the repository under `captures/YYYY/MM/DD/`.
You can also run the workflow manually from the Actions tab with the
"Fetch camera image" workflow.

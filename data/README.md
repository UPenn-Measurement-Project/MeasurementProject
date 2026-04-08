# data

## Structure
- `measurements/` -- CSV files of human-annotated femur measurements (dated snapshots, e.g., 20260407.csv). Each row is a DXA image with 20 measurements (10 per side: medial/lateral cortical thickness, shaft width, femoral head diameter, horizontal/vertical offset, neck width, hip axis length, neck axis length, neck-shaft angle).
- `box_images/` -- DXA DICOM images downloaded from Box (gitignored)
- `segmentation/` -- Segmentation masks and X-ray images for U-Net training (gitignored)
- `webscrape_images/` -- Script for web scraping additional DXA images

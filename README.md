# Python Vasculature BioMarker toolbox documentation
    
## Description

Digital fundus images are specialized photographs of the interior surface of the eye, capturing detailed views of the retina including blood vessels, optic disk and macula. They are invaluable tools in ophthalmology and optometry for diagnosing and monitoring various ocular diseases, including diabetic retinopathy, glaucoma, and macular degeneration.

The fundus image allows for the visualization of numerous vascular features, notably the arterioles (small arteries) and venules (small veins). Advanced image processing and machine learning techniques now enable the extraction of arterioles and venules from fundus images, a process known as A/V segmentation. By isolating these vessels, we can examine their morphology and distribution in greater detail, revealing subtle changes that might otherwise go unnoticed.

From this A/V segmentation, we can compute vasculature biomarkers, which are quantifiable indicators of biological states or conditions. By analyzing these vasculature biomarkers, healthcare professionals can gain deeper insights into a patient's ocular health and potentially detect early signs of disease. This approach represents a promising frontier in eye care, allowing for more proactive and personalized treatment strategies.

Eleven biomarkers have been engineered independently on the arterioles and venules segmentation, namely: 

* Area: This refers to the density of the blood vessels calculated as the total number of pixels in the segmentation, and is expressed in square pixels (:math:`pixels^2`).

* Length: This represents the cumulative length of the blood vessel derived from the segmentation. It is computed as the necessary distance to traverse the entire segmentation and is expressed in pixels.

* Perimeter: This is the total perimeter of the blood vessel from a segmentation, calculated as the required distance to traverse the outer boundary of the segmentation, and is expressed in pixels.

* Number of Endpoints: This refers to the count of points in the segmentation that correspond to the termination of a blood vessel.

* Number of Intersection Points: This is the count of points in the segmentation that correspond to an intersection within a blood vessel.

* Median Tortuosity: This is the median value of the tortuosity distribution for all blood vessels, computed using the arc-chord ratio.

* Median Branching Angle: This is the median value of the branching angle distribution for all blood vessels, and it is expressed in degrees (°).

* Capacity Dimension: D0 (also known as the box-counting dimension) is a measure of the space-filling capacity of the pattern.

* Entropy Dimension: D1 (also known as the entropy dimension) is a measure of the distribution of the pattern.

* Correlation Dimension: D2 (also known as the correlation dimension) is a measure of the correlation of the pattern.

* Singularity Length: SL represents the range of fluctuation in the fractal dimension, providing information about the complexity of local variations in the image.


## Installation

Available on pip. Run: "pip install pvbm"

pip project: https://pypi.org/project/pvbm/

read the docs: https://pvbm.readthedocs.io/en/latest/

github: https://github.com/aim-lab/PVBM


Official implementation, based on the work published by Fhima, Jonathan, Jan Van Eijgen, Ingeborg Stalmans, Yevgeniy Men, Moti Freiman, and Joachim A. Behar. “PVBM: A Python Vasculature Biomarker Toolbox Based on Retinal Blood Vessel Segmentation.” In Computer Vision–ECCV 2022 Workshops: Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part III, pp. 296-312. Cham: Springer Nature Switzerland, 2023 (https://link.springer.com/chapter/10.1007/978-3-031-25066-8_15).

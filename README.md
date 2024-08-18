# Exploring the Influence of Sea Surface Temperature on Hurricane Wind Speed Trends in the Atlantic Ocean
This project investigates the correlation between Sea Surface Temperature (SST) anomalies and hurricane maximum subtended Wind Speed (WS) trends in the North Atlantic Ocean, covering data from 1850 to 2023. The primary objective is to determine whether increasing SST trends can be reliably attributed to rising hurricane intensities, particularly in the context of climate change.

## Key Features

- **Data Sources**: Utilizes SST anomaly data from the Met Office Hadley Centre (HadSST.4.0.0.0) and hurricane wind speed data from the National Hurricane Center (NHC, HURDAT2 dataset).
- **Statistical Analysis**: Performs Pearson correlation analysis to quantify the relationship between SST anomalies and hurricane wind speeds.
- **Trend Analysis**: Analyzes long-term trends in both SST and WS using polynomial fitting and compares trends across different regions (Northern Hemisphere and Global).
- **Climatic Oscillations**: Considers the impact of oscillatory climate patterns like the North Atlantic Oscillation (NAO) and El Niño Southern Oscillation (ENSO) on SST and WS trends.
- **Fourier Analysis**: Includes a Fourier analysis of WS data to identify any significant frequencies that may indicate underlying oscillatory patterns.

## Project Structure

- **`data/`**: Contains the datasets used for analysis, including SST anomalies and hurricane wind speeds.
- **`scripts/`**: Python scripts for data processing, analysis, and visualization.
- **`notebooks/`**: Jupyter notebooks that provide step-by-step documentation of the data analysis process.
- **`results/`**: Outputs of the analysis, including plots, correlation coefficients, and summary tables.
- **`README.md`**: Project documentation.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/SST-Hurricane-Wind-Speed-Project.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd SST-Hurricane-Wind-Speed-Project
    ```
3. **Set up a virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing**:
   - Use the scripts in the `scripts/` directory to preprocess the SST and WS datasets. This includes handling missing data and calculating uncertainty bounds.

2. **Run Analysis**:
   - Execute the Python scripts or run the Jupyter notebooks in `notebooks/` to perform the correlation and trend analysis.

3. **Visualization**:
   - Generate plots and visualizations of SST and WS trends, as well as the results of the correlation analysis.

## Results

- **Correlation Findings**: The Pearson correlation coefficient between global SST anomalies and maximum hurricane WS was found to be r = 0.40, indicating a moderate positive correlation.
- **Trend Insights**: SST anomalies and WS in the North Atlantic have both shown increasing trends over the period studied, with global SST variations playing a significant role in hurricane intensity.
- **Climatic Oscillations**: The study found that external climatic factors significantly influence the SST and WS trends, making it challenging to attribute these trends solely to direct SST increases.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or enhancements.

## License

This project and report have been submitted to The University of Glasgow for assessment, meaning copying literal sentences from the technical report will result in plagiarism! Don't worry, you can use the code freely.

## Contact

Should you have any questions contact me at anaparraprado@gmail.com  Good luck! :)

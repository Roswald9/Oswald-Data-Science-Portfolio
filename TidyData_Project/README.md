# Portfolio Update 2 - Tidy Data Project - Rowan Oswald

## Project Overview

The goal of this project is to apply tidy data principles to the
`olympics_08_medalists.csv` dataset and perform exploratory data analysis
(EDA) on the 2008 Olympic medalists.

The original dataset is structured in a wide format, with separate columns
representing combinations of gender and sport. This structure makes it
more difficult to compare medal distributions and aggregate data across
categories.

Tidy data follows three main principles:

1. Each variable forms its own column.
2. Each observation forms its own row.
3. Each type of observational unit forms its own table.

The notebook uses pandas to reshape and clean the dataset into a tidy
format. Visualizations and pivot tables are then used to explore medal
distributions by medal type, sport, and gender.

## Instructions

### Requirements and Dependencies

This project requires Python and the following libraries:

- pandas: Data manipulation and cleaning
- matplotlib: Data visualization
- seaborn: Statistical visualizations
- jupyter: Running the notebook
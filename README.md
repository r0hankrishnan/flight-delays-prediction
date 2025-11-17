# Flight Delay Prediction at Philadelphia International Airport

## Project Overview

This repository contains the code and documentation for our class project: **Predicting Flight Delays at Philadelphia International Airport**. Our team will build a data-driven model to estimate the likelihood of flight delays, helping airport administrators optimize operations and improve efficiency.

## Team

- **Arnav Chatani** 
- **Hellen Jin** 
- **Rohan Krishnan** 

## Data Sources

- **Flight Delay Data:** U.S. Bureau of Transportation Statistics ([link](https://www.transtats.bts.gov/tables.asp?QO_VQ=EFD&QO_anzr=Nv4yv0r)), multi-year records for Philadelphia International Airport.
    - Currently using kaggle data from 2009-2013 as a placeholder ([link](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018))
- **Weather Data:** OpenWeather API ([link](https://openweathermap.org/api/one-call-3#history_daily_aggregation)), daily weather conditions matched to flight dates.
- **Airport Location Data**:

## Objective

Acting as data science consultants, our goal is to build a predictive model that flags flights at high risk of delay. This will help Philadelphia International Airport administrators optimize passenger flow and resource allocation, especially in light of recent FAA funding reductions and the significant costs associated with delays.

## Modeling Approach

We will explore several classification models, including:
    - TBD

The target variable will be an indicator for meaningful delays, engineered from the `arr_delay` column.

## Anticipated Challenges

- Managing and cleaning large, multi-year datasets.
- Integrating weather data via API calls and joining with flight records.
- Addressing inconsistencies in government-aggregated data.

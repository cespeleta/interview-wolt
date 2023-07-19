# Demand Forecasting Data Science Challenge

## Business Introduction

At Wolt we deliver what our customers want. We started with delivering food from restaurants and expanded to pharmacies, flower shops, supermarkets, and more.

We call places where our customers order from venues and couriers are then delivering those goods to our customers.

## Business Problem Description

We are obsessed with best-in-class customer experience. That starts with a wide variety of venues, on-time delivery, and great customer support in case something didn’t go as planned (e.g. the staff at the venue forgot to add a beverage).

On-time delivery is primarily driven by how many couriers are available to handle the number of orders.

Similarly, excellent customer support means low response times. Hence we want to schedule sufficient staff answering customer requests.

Ultimately, both are caused by the number of orders. We want to inform couriers if we’re expecting higher demand and simultaneously, we schedule sufficient support agents.

When the provided demand forecast isn’t accurate both over-supplying and under-supplying can have costly consequences, e.g.:
- We might staff too many support agents
- Or we staff too few support agents and lose valuable customers through bad customer experience.

## Required Task

We ask you to help us out with the problem of forecasting the daily number of orders. We have the daily number of (simulated) orders for Berlin, Germany from 2nd May 2020 until 30 June 2022. In addition, we have collected data for the temperature and marketing spend.

## Business requirements

- We are required to run the model every Monday morning and get predictions for the upcoming 2 weeks. Note that the values in the temperature column are real values, hence are just available for current and past days. On the other hand, marketing spend is a feature we can know in advance, so it is available at prediction time.
- Moreover, we are not just interested in the point predictions but also in the uncertainty of the predictions.
- Before putting this model into production, we need to assess model performance, so please provide information about the out-of-sample-expected performance.
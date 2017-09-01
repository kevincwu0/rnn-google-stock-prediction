# rnn-google-stock-prediction

Recurrent Neural Networks for predicting Times Series

Stanford Research Paper Times Series Prediction with Recurrent Neural Networks to predict Google Stock Price. Comparable to even substantially better than Stanford's results in 2012 with LSTM, substantial research (10 days just researching) - did not believe the results as they were too good to be true, duty of care to disclose. Not possible to predict Stock Market to that extent, investigating the results, ideally it's impossible (100%) or results shown.

What we suspect is going on here: real Google Stock price is in red and the predicted tcok price is in blue, should be shifted to the right and the RNN is taking the current amount and adding a little amount it percentage, and that's its prediction. Copy what happened today with a slight adjustment, bad news is that the RNN can't learn from the price as there's too many movement and cannot learn certain patterns and predict what's going on, pretty much just seeing what's it see today and adding a small value upward trend, not predict and not seem possible. Not possible with this kind of implementation. Good news is that all of the code is still valid, exactly how you'd construct an LSTM, how you'd add layers, how to tune it, how to add parameters and sets input for LSTM, still applicable. 

THe code is applicable. Sales pattern or volumes of customers still be able to apply the same template and check results -> shift by one. Unfortunately, Google stock price is not working across the board - fall into the same pitfall. 

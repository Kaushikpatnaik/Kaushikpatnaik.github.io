Paper Notes on - "Joint Modeling of Event Sequence and Time Series with Attentional Twin Recurrent Neural Networks"

1. What problem is the paper solving. 

A lot of data in the world can be thought of as being produced from a system over time. That is we collect a lot of information around the system/process - both in terms of system observations, environment observations and events in that system. A natural question for such systems, is whether we can accurately predict events in future: both their type and the time. 

2. What is the general idea they are using to solve the problem 

This paper attempts to answer such questions by modelling the conditional probability of next event (given all the past observations) as a non-linear mapping of two RNNs: one for past event observations and another for past time series observations. To improve interpretability, they also introduce an attentional mechanism over the RNNs.

3. What is the model and the experiments they are running

Event sequence - map it onto an embedding layer. The embedding layer is then passed onto an LSTM. The LSTM outputs are then passed onto an neural attentional layer.
Time 

3.What kind of results are they getting 
4. What previous work are they building on 
5. What is the model and the experiments they are running. Does it match the setup of the problem. Any obvious drawbacks 
6. What are some of the key proofs and assumptions on the data. Any drawbacks from real world application of the model. 
7. Starting with raw data, what steps does one have to do to recreate the results.
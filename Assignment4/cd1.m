function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    visible_data = sample_bernoulli(visible_data);
    v1  = visible_data;
    h1  = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, v1));
    v2  = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, h1));
    %h2  = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, v2));
    h2  = visible_state_to_hidden_probabilities(rbm_w, v2);
    ret = configuration_goodness_gradient(h1, v1) - ...
          configuration_goodness_gradient(h2, v2);
    ret = ret';
end

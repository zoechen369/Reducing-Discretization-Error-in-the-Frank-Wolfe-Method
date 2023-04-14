function y = rk44(x,t,step_fn);

k1 = step_fn(t,x);
k2 = step_fn(t+1/2,x+k1/2);
k3 = step_fn(t+1/2,x+k2/2);
k4 = step_fn(t+1,x+k3);

y = (k1+2*k2+2*k3+k4)/6;
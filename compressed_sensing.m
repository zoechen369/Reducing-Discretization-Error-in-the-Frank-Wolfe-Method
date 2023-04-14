clc
clear

n = 500;
m = 500;
splevel = .1;
noiselevel = 0.0;
alpha = 50;

x0 = rand(n,1) < splevel;
x0 = x0.*randn(n,1);

A = randn(m,n);
b = A*x0 + noiselevel*randn(m,1);

fn = @(x)(sum((A*x-b).^2)/m/2);
AA = A'*A;
Ab = A'*b;
grad_fn = @(x)((AA*x-Ab)/m);



maxiter = 10000;


gap = zeros(maxiter,3);
disc_err = zeros(maxiter,3);
recv_err = zeros(maxiter,3);


%% Vanilla Frank Wolfe

x = zeros(n,1);
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;

    gap(iter,1) = -z'*(x-s);
    disc_err(iter,1) = norm(s-x);
    recv_err(iter,1) = sum((x==0)~=(x0==0));
    x = x + gamma*(s-x);
end
x1 = x;


%%  Frank Wolfe RK44

stepfn = @(t,x)(2/(2+t)*(alpha*get_lmo(-grad_fn(x))-x));
x = zeros(n,1);
for iter = 1:(maxiter/4)
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;



    idx = (iter-1)*4+(1:4);
    gap(idx,2) = -z'*(x-s);
    disc_err(idx,2) = norm(s-x);
    recv_err(idx,2) = sum((x==0)~=(x0==0));


    step = rk44(x,iter,stepfn)/gamma;
        x = x + gamma*step;

end
x2 = x;



%%  Averaged Frank Wolfe 

x = zeros(n,1);
savg = x;
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;
    savg = savg + gamma*(s-savg);

    gap(iter,3) = -z'*(x-s);
    disc_err(iter,3) = norm(savg-x);
    recv_err(iter,3) = sum((x==0)~=(x0==0));


    step = gamma*(savg-x);

    x = x + step;

end


%%  AvgGrad

x = zeros(n,1);
gcum = x;
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    gcum = gcum + z;
    s = get_lmo(z)*alpha;

    gap(iter,4) = -z'*(x-s);
    disc_err(iter,4) = norm(savg-x);
    recv_err(iter,4) = sum((x==0)~=(x0==0));


    s = get_lmo(gcum)*alpha;
    x = x + gamma*(s-x);

end
x4 = x;


%%  FW+LS



x = zeros(n,1);
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;

    gap(iter,5) = -z'*(x-s);
    disc_err(iter,5) = norm(s-x);
    recv_err(iter,5) = sum((x==0)~=(x0==0));
    step =  gamma*(s-x);

     fbest = inf;
    for ss = linspace(0,1,25)
        xc = x + ss * step;
        fss = fn(xc);
        if fss < fbest
            fbest = fss;
            xbest = xc;
        end
    end
    x = xbest;
end
x1 = x;


%%  RKFW+LS

stepfn = @(t,x)(2/(2+t)*(alpha*get_lmo(-grad_fn(x))-x));
x = zeros(n,1);
for iter = 1:(maxiter/4)
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;



    idx = (iter-1)*4+(1:4);
    gap(idx,6) = -z'*(x-s);
    disc_err(idx,6) = norm(s-x);
    recv_err(idx,6) = sum((x==0)~=(x0==0));


    step = rk44(x,iter,stepfn);
     fbest = inf;
    for ss = linspace(0,1,25)
        xc = x + ss * step;
        fss = fn(xc);
        if fss < fbest
            fbest = fss;
            xbest = xc;
        end
    end
    x = xbest;

end
x2 = x;




%%  AvgFW+LS
x = zeros(n,1);
savg = x;
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;
    savg = savg + gamma*(s-savg);

    gap(iter,7) = -z'*(x-s);
    disc_err(iter,7) = norm(savg-x);
    recv_err(iter,7) = sum((x==0)~=(x0==0));


    step = gamma*(savg-x);


    fbest = inf;
    for ss = linspace(0,1,25)
        xc = x + ss * step;
        fss = fn(xc);
        if fss < fbest
            fbest = fss;
            xbest = xc;
        end
    end
x = xbest;
%     x = x + gamma*step;

end





%%  AvgGrad+LS

x = zeros(n,1);
gcum = x;
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    gcum = gcum + z;
    s = get_lmo(z)*alpha;

    gap(iter,8) = -z'*(x-s);
    disc_err(iter,8) = norm(savg-x);
    recv_err(iter,8) = sum((x==0)~=(x0==0));


    s = get_lmo(gcum)*alpha;
    
step = gamma*(s-x);

 fbest = inf;
    for ss = linspace(0,1,25)
        xc = x + ss * step;
        fss = fn(xc);
        if fss < fbest
            fbest = fss;
            xbest = xc;
        end
    end
x = xbest;

end
x4 = x;




%%  AwayStep (LS)

x = zeros(n,1);
slist = [];
glist = [];
for iter = 1:maxiter
    gamma = 2/(2+iter);
    z = -grad_fn(x);
    s = get_lmo(z)*alpha;
    slist = [slist,s];
glist = [glist,gamma];
[~,j] = min(s'*z);
daway = x - s(:,j);



    gap(iter,9) = -z'*(x-s);
    disc_err(iter,9) = norm(savg-x);
    recv_err(iter,9) = sum((x==0)~=(x0==0));


    if z'*(s-x) >= z'*daway
    step = gamma * (s-x);
    else
        step = daway*glist(j)/(1-glist(j));
    end

     % step size search
    fbest = inf;
    for ss = linspace(0,1,25)
        xc = x + ss * step;
        fss = fn(xc);
        if fss < fbest
            fbest = fss;
            xbest = xc;
        end
    end
    x  = xbest;
end
x5 = x;

%%

figure(1)
clf
subplot(1,3,1)
loglog(gap(:,1:4),'linewidth',2)
hold on
set(gca,'ColorOrderIndex',1)
loglog(gap(:,5:9),'linewidth',2,'linestyle','--')
hold on
plot(1:maxiter,8000./(1:maxiter),'k')
plot(1:maxiter,10000./(1:maxiter).^(3/2),'k--')
xlabel('gradient calls')
ylabel('gap')

ylim([.01,1e4])
subplot(1,3,2)

semilogx(recv_err(:,1:4),'linewidth',2)
hold on
set(gca,'ColorOrderIndex',1)
semilogx(recv_err(:,5:9),'linewidth',2,'linestyle','--')
hold on
% legend('FW','RK44-FW','AvgFW','AvgGrad','FW+LS','RKFW+LS','AvgFW+LS','AvgGrad+LS','AwayStep (LS)','O(1/k)','O(1/k^{1.5})',...
%     location='eastoutside')

xlabel('gradient calls')
ylabel('sparse recovery error')
subplot(1,3,3)

semilogx(recv_err(:,1:4)*nan,'linewidth',2)
hold on
set(gca,'ColorOrderIndex',1)
semilogx(recv_err(:,5:9)*nan,'linewidth',2,'linestyle','--')
hold on
axis off
legend('FW','RK44-FW','AvgFW','AvgGrad','FW+LS','RKFW+LS','AvgFW+LS',...
    'AvgGrad+LS','AwayStep (LS)','O(1/k)','O(1/k^{1.5})')

% legend('FW','RK44-FW','AvgFW')


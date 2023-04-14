function s = get_lmo(z)
s = z*0;
[~,i] = max(abs(z));
s(i) = sign(z(i));

end
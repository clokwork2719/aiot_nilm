$$
\begin{aligned}
&\text{1) } h_1(x_t) = \alpha x_t \text{ , } \alpha = \text{random}(0.1, 0.8); \\
\\
&\text{2) } h_2(x_t) = \beta_t x_t \\
&\phantom{\text{2) } h_2(x_t) = } \beta_t = 
\begin{cases} 
0 & \text{start}-\text{time} < t < \text{end}-\text{time} \\ 
1 & \text{else} 
\end{cases} \\
\\
&\text{3) } h_3(x_t) = \gamma_t x_t \ \gamma_t = \text{random}(0.1, 0.8); \\
\\
&\text{4) } h_4(x_t) = \gamma_t \ \text{mean}(\mathbf{x}) \text{, } \gamma_t = \text{random}(0.1, 0.8); \\
\\
&\text{5) } h_5(x_t) = \text{mean}(\mathbf{x}); \\
\\
&\text{6) } h_6(x_t) = x_{24-t}.
\end{aligned}
$$
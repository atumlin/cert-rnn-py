function gen_fixtures()
% Generate cross-validation fixtures: run CertRNN on fixed inputs and save
% (c, V) so the Python port can be compared numerically.
%
% Run:  matlab -batch "cd('/home/verivital/Anne/dev/cert-rnn-py/tests/matlab'); gen_fixtures"
%
% Each fixture stores:
%   - inputs (c_in/V_in for unary; x_c/x_V/y_c/y_V for bilinear; weights+state for lstm_step)
%   - outputs (c_out/V_out, or h_c/h_V/c_c/c_V)
% Python's column order matches MATLAB's because both append fresh
% predicates at the END of V; pred_ids in the Python port are only used
% for Minkowski alignment, which is a no-op when all inputs share a prefix.

    this_dir = fileparts(mfilename('fullpath'));
    nnv_dir = '/home/verivital/Anne/dev/nnv3-cert-rnn/code/nnv';
    saved_cwd = pwd;
    cd(nnv_dir);
    startup_nnv;
    cd(saved_cwd);

    fix_dir = fullfile(this_dir, 'fixtures');
    if ~isfolder(fix_dir), mkdir(fix_dir); end

    % ----- unary: tanh, sigmoid -----
    unary = {
        'tanh_1d_pos',       1.0,         0.5,                                       @(z) CertRNN.tanhZono(z)
        'tanh_1d_wide',      0.0,         3.0,                                       @(z) CertRNN.tanhZono(z)
        'tanh_1d_saturated', 4.5,         1.5,                                       @(z) CertRNN.tanhZono(z)
        'tanh_2d_shared',    [0.5; -0.3], [0.4 0.2 0.1; 0.1 0.3 0.2],                @(z) CertRNN.tanhZono(z)
        'sigmoid_1d_pos',    1.0,         0.5,                                       @(z) CertRNN.sigmoidZono(z)
        'sigmoid_1d_wide',   0.0,         3.0,                                       @(z) CertRNN.sigmoidZono(z)
        'sigmoid_2d_shared', [0.5; -0.3], [0.4 0.2 0.1; 0.1 0.3 0.2],                @(z) CertRNN.sigmoidZono(z)
    };
    for i = 1:size(unary, 1)
        name = unary{i, 1};
        c_in = unary{i, 2};
        V_in = unary{i, 3};
        fn   = unary{i, 4};
        z_in  = Zono(c_in, V_in);
        z_out = fn(z_in);
        c_out = z_out.c;
        V_out = z_out.V;
        save(fullfile(fix_dir, [name '.mat']), 'c_in', 'V_in', 'c_out', 'V_out');
        clear c_in V_in c_out V_out;
        fprintf('  wrote %s.mat\n', name);
    end

    % ----- bilinear: sigid (x * sigma(y)) -----
    sigid = {
        'sigid_1d_C1',   1.0,  1.0,   0.0, 1.0
        'sigid_1d_C2',  -1.0,  1.0,   0.0, 1.0
        'sigid_1d_C3',   0.0,  1.0,   0.0, 1.0
        'sigid_1d_asym',-0.5,  1.5,   0.0, 1.0
    };
    for i = 1:size(sigid, 1)
        name = sigid{i, 1};
        z_x = Zono(sigid{i, 2}, sigid{i, 3});
        z_y = Zono(sigid{i, 4}, sigid{i, 5});
        z_out = CertRNN.bilinearSigmoidIdentity(z_x, z_y);
        x_c = z_x.c; x_V = z_x.V;
        y_c = z_y.c; y_V = z_y.V;
        c_out = z_out.c; V_out = z_out.V;
        save(fullfile(fix_dir, [name '.mat']), 'x_c', 'x_V', 'y_c', 'y_V', 'c_out', 'V_out');
        clear x_c x_V y_c y_V c_out V_out;
        fprintf('  wrote %s.mat\n', name);
    end

    % ----- bilinear: sigtanh (sigma(x) * tanh(y)) -----
    sigtanh = {
        'sigtanh_1d_pos',   1.0,  1.0,    1.0, 1.0
        'sigtanh_1d_neg',  -1.0,  1.0,   -1.0, 1.0
        'sigtanh_1d_mixed', 0.0,  1.0,    0.0, 1.0
        'sigtanh_1d_wide',  0.0,  3.0,    0.0, 3.0
        'sigtanh_1d_asym', -0.5,  1.5,    0.5, 1.5
    };
    for i = 1:size(sigtanh, 1)
        name = sigtanh{i, 1};
        z_x = Zono(sigtanh{i, 2}, sigtanh{i, 3});
        z_y = Zono(sigtanh{i, 4}, sigtanh{i, 5});
        z_out = CertRNN.bilinearSigmoidTanh(z_x, z_y);
        x_c = z_x.c; x_V = z_x.V;
        y_c = z_y.c; y_V = z_y.V;
        c_out = z_out.c; V_out = z_out.V;
        save(fullfile(fix_dir, [name '.mat']), 'x_c', 'x_V', 'y_c', 'y_V', 'c_out', 'V_out');
        clear x_c x_V y_c y_V c_out V_out;
        fprintf('  wrote %s.mat\n', name);
    end

    % ----- lstm_step: tiny deterministic model, T=1 and T=3 single-frame -----
    rng(20260514);
    D = 2; H = 2;
    W_in  = 0.5 * randn(4*H, D);
    W_rec = 0.3 * randn(4*H, H);
    b     = 0.1 * randn(4*H, 1);
    mu    = [0.5; -0.3];
    eps_v = 0.2;
    x_c = mu;
    x_V = eps_v * eye(D);

    z_x  = Zono(mu, eps_v * eye(D));
    z_h0 = Zono(zeros(H, 1), zeros(H, 0));
    z_c0 = Zono(zeros(H, 1), zeros(H, 0));

    [z_h1, z_c1] = CertRNN.lstmStep(z_x, z_h0, z_c0, W_in, W_rec, b);
    h_c = z_h1.c; h_V = z_h1.V; c_c = z_c1.c; c_V = z_c1.V;
    save(fullfile(fix_dir, 'lstm_step_1step.mat'), ...
        'D', 'H', 'W_in', 'W_rec', 'b', 'x_c', 'x_V', 'h_c', 'h_V', 'c_c', 'c_V');
    fprintf('  wrote lstm_step_1step.mat\n');

    [z_h2, z_c2] = CertRNN.lstmStep(z_x, z_h1, z_c1, W_in, W_rec, b);
    [z_h3, z_c3] = CertRNN.lstmStep(z_x, z_h2, z_c2, W_in, W_rec, b);
    h_c = z_h3.c; h_V = z_h3.V; c_c = z_c3.c; c_V = z_c3.V;
    T = 3;
    save(fullfile(fix_dir, 'lstm_step_3step.mat'), ...
        'D', 'H', 'T', 'W_in', 'W_rec', 'b', 'x_c', 'x_V', 'h_c', 'h_V', 'c_c', 'c_V');
    fprintf('  wrote lstm_step_3step.mat\n');

    % ----- rnn_step: T=5 single-frame -----
    rng(20260515);
    D = 2; H = 3;
    W_in  = 0.5 * randn(H, D);
    W_rec = 0.3 * randn(H, H);
    b     = 0.1 * randn(H, 1);
    mu    = [0.5; -0.3];
    eps_v = 0.2;
    x_c = mu;
    x_V = eps_v * eye(D);

    z_x  = Zono(mu, eps_v * eye(D));
    z_h  = Zono(zeros(H, 1), zeros(H, 0));
    T = 5;
    for t = 1:T
        z_h = CertRNN.rnnStep(z_x, z_h, W_in, W_rec, b);
    end
    h_c = z_h.c; h_V = z_h.V;
    save(fullfile(fix_dir, 'rnn_step_5step.mat'), ...
        'D', 'H', 'T', 'W_in', 'W_rec', 'b', 'x_c', 'x_V', 'h_c', 'h_V');
    fprintf('  wrote rnn_step_5step.mat\n');

    fprintf('\nAll fixtures saved to %s\n', fix_dir);
end

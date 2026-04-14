%% WSN Dataset Simulation
% Generates dataset for clone detection experiment

clc
clear
close all

%% PARAMETERS

NUM_NODES = 100;
NUM_ROUNDS = 20;

CLONE_PERCENT = 0.12;
MALICIOUS_PERCENT = 0.05;

rows = [];

%% SIMULATION

for r = 1:NUM_ROUNDS

    for node = 1:NUM_NODES

        label = 0;

        rand_val = rand;

        if rand_val < CLONE_PERCENT
            label = 1;
        elseif rand_val < CLONE_PERCENT + MALICIOUS_PERCENT
            label = 2;
        end

        %% Packet rate model

        if label == 0
            packet_rate = 28 + 5*randn;
        elseif label == 1
            packet_rate = 45 + 6*randn;
        else
            packet_rate = 55 + 8*randn;
        end

        packet_rate = max(5, packet_rate);

        %% Energy model

        energy_remaining = max(0.1, 1 - r*0.015 + 0.03*randn);

        energy_consumed = 150 + (450-150)*rand;

        %% Network topology

        x = rand*100;
        y = rand*100;

        dist = 5 + (60-5)*rand;

        is_ch = rand < 0.1;

        rows = [rows;
            node, r, packet_rate, energy_remaining, ...
            energy_consumed, dist, is_ch, x, y, label];

        %% Clone injection

        if label == 1 && rand < 0.9

            clone_x = rand*100;
            clone_y = rand*100;

            clone_packet = packet_rate + 5*randn;

            clone_energy = max(0.1, energy_remaining - 0.03*randn);

            rows = [rows;
                node, r, clone_packet, clone_energy, ...
                energy_consumed, dist, is_ch, clone_x, clone_y, label];
        end

    end
end

%% CREATE TABLE

data = array2table(rows, 'VariableNames', { ...
    'node_id',...
    'round',...
    'packet_rate',...
    'energy_remaining',...
    'energy_consumed_uJ',...
    'dist_to_ch_bs',...
    'is_cluster_head',...
    'x_pos',...
    'y_pos',...
    'label'});

%% SAVE CSV

if ~exist('data','dir')
    mkdir('data')
end

writetable(data,'data/wsn_data.csv');

disp("WSN dataset generated successfully")
disp("Total records:")
disp(height(data))
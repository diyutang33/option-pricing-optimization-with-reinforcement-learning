# CMPS 140 Winter 2019
# SPY Stock Options Pricing with Reinforcement Learning
# Randall Li (rhli@ucsc.edu)
# Di Yu Tang (dtang10@ucsc.edu)
# Python 3

import csv
import matplotlib.pyplot as plt
import random
import math
import operator

date_list = []
option_chain = {}
q_table = {}
stock_price = {}
alpha = 0.5
gamma = 0.5  # More optimal, takes much longer to learn. Higher gamma the better, learns faster
q_total = 0

with open('twomonths_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    line_count = 0
    for row in csv_reader:
        if row[0] != '':
            line_count += 1
            if str(row[0]) not in stock_price:
                stock_price[str(row[0])] = float(row[1])
            if str(row[0]) not in option_chain:
                option_chain[str(row[0])] = {}
            if str(row[2]) not in option_chain[str(row[0])]:
                option_chain[str(row[0])][str(row[2])] = {}
            if str(row[3]) not in option_chain[str(row[0])][str(row[2])]:
                option_chain[str(row[0])][str(row[2])][str(row[3])] = {}
            if str(row[4]) not in option_chain[str(row[0])][str(row[2])][str(row[3])]:
                option_chain[str(row[0])][str(row[2])][str(row[3])][str(row[4])] = float(row[5])
            if row[0] not in date_list:
                date_list.append(row[0])


# for d in option_chain:
#     for t in option_chain[d]:
#         for e in option_chain[d][t]:
#             for s in option_chain[d][t][e]:
#                 print(d, t, e, s, option_chain[d][t][e][s])
#
def date_convert(date):
    d_split = date.split('/')
    year = d_split[2]
    month = d_split[0]
    if len(month) < 2:
        month = '0' + month
    day = d_split[1]
    if len(day) < 2:
        day = '0' + day
    return int(year + month + day)


def add_q_table_state(date, position, addition_actions):
    if (date, position) not in q_table:
        q_table[(date, position)] = {}
        for t in option_chain[date]:
            for e in option_chain[date][t]:
                for s in option_chain[date][t][e]:
                    q_table[(date, position)][('buy', e, s, t)] = 0
        if len(addition_actions) > 0:  # e.g. closing position
            for a in addition_actions:
                exp_num = date_convert(a[1])
                date_num = date_convert(date)
                if date_num < exp_num:
                    q_table[(date, position)][a] = 0
    return (date, position)


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        random_a = random.choice(list(q_table[state].keys()))
        while random_a in banned_actions:
            random_a = random.choice(list(q_table[state].keys()))
        return random_a
    else:
        max_sell_action_list = []  # to encourage closing
        for a in q_table[state]:
            if 'sell' in a and q_table[state][a] == max(q_table[state].values()) and a not in banned_actions:
                max_sell_action_list.append(a)
        if len(max_sell_action_list) > 0:
            return random.choice(max_sell_action_list)
        else:
            k = 0
            sorted_q = sorted(q_table[state].items(), key=operator.itemgetter(1), reverse=True)
            max_q_buy = sorted_q[k][0]
            while max_q_buy in banned_actions:
                k += 1
                max_q_buy = sorted_q[k][0]
            return max_q_buy

episode_trade_history = []
optimal_trade_history = []
expired_history = []
q_history = {'episode': [], 'total_q': [], 'avg_q': []}

add_q_table_state(date_list[0], (), [])
done = False
episode_i = 1
current_q_list = []  # convergence check
epsilon_list = []

while not done:
    # epsilon = 1/episode_i # may leave some tail Q-values 0
    epsilon = 1 / (math.sqrt(episode_i))
    state = (date_list[0], ())
    position_price = {}
    current_position = []
    addition_actions = []
    banned_actions = []
    new_q_list = []  # convergence check
    optimal_trade_history = []  # refreshes every episode
    expired_history = []
    j = 0
    while j < len(date_list) - 1:
        reward = 0
        action = choose_action(state)
        date = date_list[j]
        exp = action[1]
        strike = action[2]
        type = action[3]
        # checking for expired options
        for c in current_position:
            c_exp_num = date_convert(c[0])
            c_date_num = date_convert(date)
            if c_exp_num < c_date_num:
                current_position.remove(c)
                c_action = ('buy', c[0], c[1], c[2])
                reward += (position_price[c_action] * -100)
            if c_exp_num <= c_date_num:
                banned_sell = ('sell', c[0], c[1], c[2])
                if banned_sell not in banned_actions:
                    banned_actions.append(('sell', c[0], c[1], c[2]))

        if 'buy' in action:
            current_position.append((exp, strike, type))
            position_price[action] = option_chain[date][type][exp][strike]
            add_sell = ('sell', exp, strike, type)
            if add_sell not in banned_actions:
                addition_actions.append(('sell', exp, strike, type))
            banned_actions.append(('buy', exp, strike, type))
            next_state = add_q_table_state(date_list[j + 1], tuple(current_position), addition_actions)
        elif 'sell' in action:
            current_position.remove((exp, strike, type))
            buy_action = ('buy', exp, strike, type)
            try:
                reward = (option_chain[date][type][exp][strike] - position_price[buy_action]) * 100
            except KeyError:
                reward = position_price[buy_action] * -100
                expired_history.append((date, reward))
            addition_actions.remove(action)
            banned_actions.remove(('buy', exp, strike, type))
            del position_price[buy_action]
            # stay on the same day if sell
            next_state = add_q_table_state(date_list[j], tuple(current_position), addition_actions)
            j -= 1
        current_Q = q_table[state][action]
        optimal_trade_history.append((state, action, current_Q, reward))
        new_q_list.append(current_Q)
        next_max = max(q_table[next_state].values())
        q_table[state][action] = ((1 - alpha) * current_Q) + (alpha * (reward + (gamma * next_max)))
        state = next_state
        j += 1
    # if len(q_history['episode']) == 500:
    #     q_history['episode'].pop(0)
    #     q_history['total_q'].pop(0)
    # q_history = {'episode': [], 'total_q': []} # only save last 500 episodes
    q_history['episode'].append(episode_i)
    # q_history['total_q'].append(sum(current_q_list))
    # q_total += sum(current_q_list)/len(q_table)
    q_history['avg_q'].append(sum(current_q_list) / len(date_list))

    len_current_q_list = len(current_q_list)
    # last element of Q list is always 0
    if len_current_q_list == len(new_q_list) and all(
            a > 0 or a < 0 for a in current_q_list[:-1]) and len_current_q_list > 0:
        done = True
        m = 0
        while m < len(current_q_list):
            if abs(current_q_list[m] - new_q_list[m]) > 0.000001 or episode_i < 30000:
                print(epsilon)
                done = False
                m = len_current_q_list
            m += 1
            # if abs(current_q_list[i] - new_q_list[i]) <= 0.001:
    print('Episode', episode_i)
    print(current_q_list)
    print(new_q_list)
    current_q_list = new_q_list.copy()
    episode_i += 1

# optimal trade simulator -------------------

sim_balance = 0
sim_balance_history = {'date': [], 'balance': []}
sim_trade_record = {}
sim_position = {}
plot_buy_call = {'date': [], 'strike': []}
plot_sell_call = {'date': [], 'strike': []}
plot_buy_put = {'date': [], 'strike': []}
plot_sell_put = {'date': [], 'strike': []}

plot_trade_history = {}
sim_current_holding = []
print('optimal trade----------')
for t in optimal_trade_history:
    if 'buy' in t[1]:
        sim_position[t[1]] = option_chain[t[0][0]][t[1][3]][t[1][1]][t[1][2]]
        if t[1][3] == 'call':
            plot_buy_call['date'].append(t[0][0])
            plot_buy_call['strike'].append(float(t[1][2]))
        elif t[1][3] == 'put':
            plot_buy_put['date'].append(t[0][0])
            plot_buy_put['strike'].append(float(t[1][2]))
        if (t[0][0], sim_balance) not in sim_trade_record:
            sim_trade_record[(t[0][0], sim_balance)] = ''
        sim_trade_record[(t[0][0], sim_balance)] += ('Buy ' + t[1][2] + ' ' + t[1][1] + ' ' + t[1][3] + '\n')
        sim_current_holding.append((t[1][1], t[1][2], t[1][3]))
    if 'sell' in t[1]:
        try:
            sim_balance += float(option_chain[t[0][0]][t[1][3]][t[1][1]][t[1][2]] - sim_position[
                ('buy', t[1][1], t[1][2], t[1][3])]) * 100
        except KeyError:
            print('error------', t[0][0], t[1][3], t[1][1], t[1][2], '|', t[1][1], t[1][2], t[1][3])
        del sim_position[('buy', t[1][1], t[1][2], t[1][3])]
        if t[1][3] == 'call':
            plot_sell_call['date'].append(t[0][0])
            plot_sell_call['strike'].append(float(t[1][2]))
        elif t[1][3] == 'put':
            plot_sell_put['date'].append(t[0][0])
            plot_sell_put['strike'].append(float(t[1][2]))
        if (t[0][0], sim_balance) not in sim_trade_record:
            sim_trade_record[(t[0][0], sim_balance)] = ''
        sim_trade_record[(t[0][0], sim_balance)] += ('Sell ' + t[1][2] + ' ' + t[1][1] + ' ' + t[1][3] + '\n')
        sim_current_holding.remove((t[1][1], t[1][2], t[1][3]))
    for h in sim_current_holding:
        exp = date_convert(h[0])
        today = date_convert(t[0][0])
        if exp < today:
            sim_current_holding.remove(h)
            punish = (sim_position[('buy', h[0], h[1], h[2])] * -100)
            sim_balance += punish
            print('expired', punish, h)
    sim_balance_history['date'].append(t[0][0])
    sim_balance_history['balance'].append(sim_balance)
    print('date:', t[0][0], '\taction:', t[1], '\tQ:', t[2])
    print('current holding', sim_current_holding)
    print('sim_balance', sim_balance)
    print('next day---------')

for s in sim_trade_record:
    print(s, sim_trade_record[s])

print('expired options------------------')
for e in expired_history:
    print(e)

print(len(expired_history))

# for i, d in enumerate(date_list):
#     print(date_list[i], stock_price[d])

plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams.update({'font.size': 12})

fig, ax1 = plt.subplots()
ax1.plot(date_list, list(stock_price.values()), label='Stock Price', linewidth=2)
ax1.plot(plot_buy_call['date'], plot_buy_call['strike'], 'g^', label='Open Call', markersize=16)
ax1.plot(plot_buy_put['date'], plot_buy_put['strike'], 'gv', label='Open Put', markersize=16)
ax1.plot(plot_sell_call['date'], plot_sell_call['strike'], 'r^', label='sell Call', markersize=16)
ax1.plot(plot_sell_put['date'], plot_sell_put['strike'], 'rv', label='sell Put', markersize=16)
ax1.set_ylabel('Stock Price', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
for n, label in enumerate(ax1.xaxis.get_ticklabels()):
    if n % 10 != 0:
        label.set_visible(False)

ax2 = ax1.twinx()
ax2.plot(sim_balance_history['date'], sim_balance_history['balance'], 'g--', linewidth=2, label='Profit')
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_ylabel('Total Profit', fontsize=20)
leg2 = ax2.legend(loc="upper right");

plt.figure(3)
plt.plot(q_history['episode'], q_history['avg_q'])
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Total Q-value', fontsize=20)

# plt.figure(2)
# plt.plot(list(range(len(epsilon_list))), epsilon_list)
plt.show()

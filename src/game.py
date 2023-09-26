import streamlit as st
import random
import json
import os
import torch
from src.utils import BO_step, func1, func2, return_scaling1, return_scaling2, generate_xtrue1, generate_xtrue2


def init(bound: int = 10, post_init=False):
    if not post_init:
        st.session_state.win = 0
        st.session_state.clicklist = []
        st.session_state.BOlist= []
        st.session_state.initlist = []
        st.session_state.numberlist = {}
    st.session_state.tries = 0
    st.session_state.init_flag = True


def restart():
    for key in st.session_state.keys():
        if bool([x for x in [',','status'] if x in key]):
            del st.session_state[key]
    init()


def update_BO_guess(x_new_usr,y_new_usr,x_train_bo,x_train_usr,y_train_bo,y_train_usr,x_true,max_bound,bounds,min_dist,max_dist,func):

    # BO Step
    x_new_bo, y_new_bo, x_train_bo, y_train_bo = BO_step(x_train_bo, y_train_bo,x_true,bounds,min_dist,max_dist,max_bound=max_bound,func=func)

    ## Update datasets
    x_train_bo  = torch.concatenate([x_train_bo,  x_new_bo  ] )
    x_train_usr = torch.concatenate([x_train_usr, x_new_usr ] )

    y_train_bo  = torch.concatenate([y_train_bo,  y_new_bo.reshape(-1,1) ] )
    y_train_usr = torch.concatenate([y_train_usr, y_new_usr.reshape(-1,1) ] )

    st.session_state.x_train_bo = x_train_bo
    st.session_state.x_train_usr = x_train_usr
    st.session_state.y_train_bo = y_train_bo
    st.session_state.y_train_usr = y_train_usr

    x_new_bo = x_new_bo.squeeze().detach().numpy()
    st.session_state.BOlist.append(str(int(x_new_bo[0]))+','+str(int(x_new_bo[1])))


def change_icon(key, x_true, bounds, max_bound, min_dist, max_dist,x_train_bo,x_train_usr,y_train_bo,y_train_usr,func):
    st.session_state.tries+=1
    st.session_state[key+'_status']=True # indicating this location is being clicked already
    i,j = key.split(',')
    st.session_state.clicklist.append(key)
    x_new_usr = torch.tensor([int(i),int(j)]).reshape(1,2)
    f = func(x_new_usr, x_true, max_bound, min_dist, max_dist)
    st.session_state.numberlist[key]=int(f.detach().numpy())

    # run BO update
    update_BO_guess(x_new_usr,f,x_train_bo,x_train_usr,y_train_bo,y_train_usr,x_true,max_bound,bounds,min_dist,max_dist,func)


def initialize_game(max_bound, func, return_scaling, generate_xtrue, n_init=3):
    # initialize a new game
    x_true = generate_xtrue(max_bound)

    bounds = [[0]*2, [max_bound]*2]
    bounds = torch.tensor(bounds)

    # get scalinge
    min_dist, max_dist = return_scaling(x_true,max_bound)

    # get init samples
    x_train = torch.ceil(torch.rand(n_init, 2)*max_bound)
    y_train = func(x_train, x_true, max_bound, min_dist, max_dist)
    y_train = y_train.reshape(-1,1)

    # check that none of the samples equal 99, if so resample
    indices = torch.where(y_train == 99)[0]
    while indices.any():
        x_train = torch.ceil(torch.rand(n_init, 2)*max_bound)
        y_train = func(x_train, x_true, max_bound, min_dist, max_dist)
        y_train = y_train.reshape(-1,1)
        indices = torch.where(y_train == 99)[0]

    x_train_bo  = x_train
    x_train_usr = x_train

    y_train_bo  = y_train
    y_train_usr = y_train

    # change button status for intial samples
    for x,y in zip(x_train,y_train):
        x = x.detach().numpy()
        i,j = int(x[0]), int(x[1])
        key = str(i)+','+str(j)
        st.session_state[key+'_status']=True
        st.session_state.numberlist[key]=int(y)
        st.session_state.initlist.append(key)

    return x_train_bo,x_train_usr,y_train_bo,y_train_usr,x_true,bounds, min_dist, max_dist


def main(level='level1'):

    # initialize level
    if level == 'level1':
        st.write(
            '''
            # 🏴‍☠️ Treasure Hunt Game: Level 1
            '''
        )
        func = func1
        return_scaling = return_scaling1
        generate_xtrue = generate_xtrue1
        init_budget = 1
        init_n0_guess = 10

    else:
        st.write(
            '''
            # 🏴‍☠️ Treasure Hunt Game: Level 2
            '''
        )
        func = func1
        return_scaling = return_scaling1
        generate_xtrue = generate_xtrue1
        init_budget = 3
        init_n0_guess = 5

    if 'win' not in st.session_state:
        init()

    # set up settings
    reset, win, lives, settings = st.columns([0.45, 0.3, 1, 1])
    reset.button('New game', on_click=restart)
    with settings.expander('Settings'):
        max_bound = st.select_slider(
            'Set Range Max',
            options = [i for i in range(4,20)],
            value=10,
            key='bound',
            on_change=restart,
        )
        budget = st.number_input('Number of Gueses',value=init_budget, min_value=1, max_value=5)
        n0_guess = st.number_input('Initial Value Provided',value=init_n0_guess, min_value=1, max_value=15)

    # user name
    user_name = st.text_input('Player Name:','RandomPlayer1')

    # initialize game
    if st.session_state.init_flag:
        x_train_bo,x_train_usr,y_train_bo,y_train_usr,x_true,bounds,min_dist, max_dist = initialize_game(max_bound,func,return_scaling,generate_xtrue,n_init=n0_guess)
        st.session_state.x_train_bo = x_train_bo
        st.session_state.x_train_usr = x_train_usr
        st.session_state.y_train_bo = y_train_bo
        st.session_state.y_train_usr = y_train_usr
        st.session_state.x_true = x_true
        st.session_state.bounds = bounds
        st.session_state.min_dist = min_dist
        st.session_state.max_dist = max_dist
        st.session_state.init_flag = False

    # number of guesses
    st.write("Number of Guesses Used: ", st.session_state.tries)

    # retrive session data
    x_train_bo = st.session_state.x_train_bo
    x_train_usr = st.session_state.x_train_usr
    y_train_bo = st.session_state.y_train_bo
    y_train_usr = st.session_state.y_train_usr
    x_true = st.session_state.x_true
    bounds = st.session_state.bounds 
    min_dist = st.session_state.min_dist
    max_dist = st.session_state.max_dist

    # set up the playboard
    s_columns = st.columns(max_bound)
    click_button = dict()
    for i,sc in enumerate(s_columns):
        with sc:
            for j in range(max_bound):
                key = str(i+1)+','+str(j+1)
                click_button[key] = st.empty()
                # when budget used
                if st.session_state.tries==budget:
                    x_input = torch.tensor([int(i+1),int(j+1)]).reshape(1,2)
                    fvalue = func(x_input, x_true, max_bound, min_dist, max_dist)
                    fvalue = int(fvalue.detach().numpy())
                    if (key in st.session_state.clicklist) & (key in st.session_state.BOlist) :
                        click_button[key].button(':violet['+str(fvalue)+']', disabled=True, key=key)
                    elif key in st.session_state.clicklist:
                        click_button[key].button(':blue['+str(fvalue)+']', disabled=True, key=key)
                    elif key in st.session_state.BOlist:
                        click_button[key].button(':red['+str(fvalue)+']', disabled=True, key=key)
                    elif key in st.session_state.initlist:
                        click_button[key].button(':green['+str(fvalue)+']', disabled=True, key=key)
                    else:
                        click_button[key].button(str(fvalue), disabled=True, key=key)
                # when still guessing
                else:
                    if key+'_status' in st.session_state:
                        fvalue = st.session_state.numberlist[key]
                        if key in st.session_state.initlist:
                            click_button[key].button(':green['+str(fvalue)+']', disabled=True, key=key)
                        else:
                            click_button[key].button(':blue['+str(fvalue)+']', disabled=True, key=key)
                    else:
                        click_button[key].button('X', on_click=change_icon, args=[key,x_true, bounds, max_bound, min_dist, max_dist,x_train_bo,x_train_usr,y_train_bo,y_train_usr,func], key=key)

    if st.session_state.tries==budget:
        # show results
        st.info('All guesses are used!', icon="ℹ️")
        st.write('His Score is %i '%(y_train_bo.max().item()))
        st.write('Your Score is %i '%(y_train_usr.max().item()))
        if y_train_bo.max() > y_train_usr.max():
            st.error('Sorry, but Thomas Bayes won this round!', icon="🚨")
        else:
            # player wins!
            st.toast('Congragulations! You won this round!', icon='🎉')
            st.balloons()
            st.session_state.win += 1

    win.button(f'🏆 {st.session_state.win}')


if __name__ == '__main__':
    main()

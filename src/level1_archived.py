import streamlit as st
import random
import torch
import matplotlib
from matplotlib import pyplot as plt
from src.utils import BO_step, func


def init(bound: int = 100, post_init=False):
    if not post_init:
        st.session_state.input1 = 0
        st.session_state.input2 = 0
        st.session_state.win = 0
    st.session_state.tries = 0


def restart():
    init(st.session_state.bound, post_init=False)

def initialize_game(max_bound, n_init=3):
    # initialize a new game
    x_true = torch.round(torch.rand(1, 2)*max_bound)

    bounds = [[0]*2, [max_bound]*2]
    bounds = torch.tensor(bounds)

    #get init samples
    x_train = torch.round(torch.rand(n_init, 2)*100)
    y_train = func(x_train, x_true)

    x_train_bo  = x_train
    x_train_usr = x_train

    y_train_bo  = y_train
    y_train_usr = y_train

    return x_train_bo,x_train_usr,y_train_bo,y_train_usr,x_true,bounds


def generate_plot(x_train_bo,x_train_usr,y_train_bo,y_train_usr,max_bound,player_name,plot_true=False):

    # update plot
    fig, ax = plt.subplots(1,2,figsize=(18, 6))

    # Update Bayes game
    if plot_true:
        ax1 = ax[0].scatter( x_true[:,0],  x_true[:,1], marker = 'X', s=250, color='k')
        ax2 = ax[1].scatter( x_true[:,0],  x_true[:,1], marker = 'X', s=250, color='k')

    ax1 = ax[0].scatter( x_train_bo[:,0],  x_train_bo[:,1], c='k', marker = 'x', s=100,)
    ax1 = ax[0].scatter( x_train_bo[:,0]*-1,  x_train_bo[:,1], c=y_train_bo.flatten(), marker = 's', s=100, cmap='coolwarm_r', vmin=0, vmax=140)
    for i in range(len(x_train_bo)):
        ax[0].annotate(str(int(torch.round(y_train_bo[i,0]).item())), (x_train_bo[i,0].numpy(), x_train_bo[i,1].numpy() + 0.2),fontsize=12)
    ax[0].set(title=r'Thomas Bayes, $the~bandit$',xlim=(0,max_bound), ylim=(0,max_bound), )
    fig.colorbar(ax1, ax=ax[0], pad=0.1)

    #update usr game
    ax2 = ax[1].scatter(x_train_usr[:,0], x_train_usr[:,1], marker = 's', s=100, c=y_train_usr.flatten(), cmap='coolwarm_r', vmin=0, vmax=140)
    ax[1].set(title=player_name,xlim=(0,max_bound), ylim=(0,max_bound))
    for i in range(len(x_train_usr)):
        ax[1].annotate(str(int(torch.round(y_train_usr[i,0]).item())), (x_train_usr[i,0].numpy()+.2, x_train_usr[i,1].numpy() + 0.2),fontsize=12)
    fig.colorbar(ax2, ax=ax[1], pad=0.1)

    return fig



def main():

    # title of the page
    st.write(
        """
        # 🔢 Pirates Treasure Hunt: Level 1
        """
    )

    if 'win' not in st.session_state:
        init()

    reset, win, set_range = st.columns([0.39, 1, 1])
    reset.button('New game', on_click=restart)

    with set_range.expander('Settings'):
        max_bound = st.select_slider(
            'Set Range Max',
            [10*i for i in range(1, 11)],
            value=100,
            key='bound',
            on_change=restart,
        )
        budget = st.number_input('Number of Gueses',value=5, min_value=1, max_value=10)

    # initialize game
    if st.session_state.tries == 0:
        x_train_bo,x_train_usr,y_train_bo,y_train_usr,x_true,bounds = initialize_game(max_bound)
        st.session_state.x_train_bo = x_train_bo
        st.session_state.x_train_usr = x_train_usr
        st.session_state.y_train_bo = y_train_bo
        st.session_state.y_train_usr = y_train_usr
        st.session_state.x_true = x_true
        st.session_state.bounds = bounds

    user_name = st.text_input('Player Name:','RandomPlayer1')
    st.write("Number of Guesses used: ", st.session_state.tries)
    x_train_bo = st.session_state.x_train_bo
    x_train_usr = st.session_state.x_train_usr
    y_train_bo = st.session_state.y_train_bo
    y_train_usr = st.session_state.y_train_usr

    # show plot
    if st.session_state.tries==budget:
        fig = generate_plot(x_train_bo,x_train_usr,y_train_bo,y_train_usr,max_bound,user_name,plot_true=True)
    else:
        fig = generate_plot(x_train_bo,x_train_usr,y_train_bo,y_train_usr,max_bound,user_name)
    st.pyplot(fig)

    # plot initial data points:
    with st.form(key='my_form'):
        # st.write('Providing guess 1 out of 5')
        st.write('Take a look at the plot, where do you want to look next?')

        # ask user to input guess
        placeholder1, placeholder2 = st.empty(), st.empty()
        guess1 = placeholder1.number_input(
            f'Enter your guess for x1 from 1 - {max_bound}',
            key='input1',
            min_value=0,
            max_value=st.session_state.bound,
        )

        guess2 = placeholder2.number_input(
            f'Enter your guess for x2 from 1 - {max_bound}',
            key='input2',
            min_value=0,
            max_value=st.session_state.bound,
        )

        submit = st.form_submit_button('Submit Guess')

    if submit:
        st.session_state.tries +=1
        #  get new user inputs
        x_new_usr = torch.tensor([int(guess1),int(guess2)]).reshape(1,2)
        y_new_usr = func(x_new_usr, st.session_state.x_true)

        # BO Step
        x_new_bo, y_new_bo, x_train_bo, y_train_bo = BO_step(x_train_bo, y_train_bo, st.session_state.x_true,st.session_state.bounds)

        ## Update datasets
        x_train_bo  = torch.concatenate([x_train_bo,  x_new_bo  ] )
        x_train_usr = torch.concatenate([x_train_usr, x_new_usr ] )

        y_train_bo  = torch.concatenate([y_train_bo,  y_new_bo  ] )
        y_train_usr = torch.concatenate([y_train_usr, y_new_usr ] )

        st.session_state.x_train_bo = x_train_bo
        st.session_state.x_train_usr = x_train_usr
        st.session_state.y_train_bo = y_train_bo
        st.session_state.y_train_usr = y_train_usr
    
    
    if st.session_state.tries==budget:
        # show results
        st.info('All guesses are used!', icon="ℹ️")
        y_best = torch.min(y_train_bo.min(),y_train_usr.min())
        if y_train_bo.min() <= y_train_usr.min():
            st.error('Sorry, but Thomas Bayes won this round!', icon="🚨")
            st.write('He got %i steps closer'%(y_train_usr.min().item() - y_train_bo.min().item() ))

        else:
            # player wins!
            st.toast('Congragulations! You won this round!', icon='🎉')
            st.write('You got %i steps closer'%( y_train_bo.min().item() - y_train_usr.min().item() ))
            st.balloons()
            st.session_state.win += 1

    win.button(f'🏆 {st.session_state.win}')
    

if __name__ == '__main__':
    main()

import streamlit as st


def main():
    st.markdown(
        '''
        <h1 align="center">
            Welcome to the Pirates Treasure Hunt Game 👋
        </h1>

        ---

        #### About the Game

        Click the "Treasure Hunt Game" button to begin play the game! 

        Thomas Bayes, the multi-armed bandit has a special treasure locator, given enough clues, he can find the biggest treasure (💠). 
        
        Using your scientific power, you will be able to find the largest diamond too!

        Can you find the bigger 💠 than what Thomas Bayes finds?

        There are 2 levels of the game, come and try it out and see if you can beat him at both levels!
        
        ''',
        unsafe_allow_html=True,
    )

    st.image('./src/assets/Bayes.png',width=250)


if __name__ == '__main__':
    main()

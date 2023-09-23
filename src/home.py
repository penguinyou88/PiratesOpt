import streamlit as st


def main():
    st.markdown(
        '''
        <h1 align="center">
            Welcome to the Pirates Treasure Hunt Game 👋
        </h1>

        ---

        #### About

        Click the play button to begin the game!

        Thomas Bayes, the [multi-armed] bandit is has a special tresure locator, and is looking for a tresure. 
        
        Using your scientific prowess, you were able to make this tresure locatoator too!

        Can you reach the tresure before Thomas Bayes does?

        There are 2 levels of the game, come and try it out and see if you can beat him at both levels!
        ''',
        unsafe_allow_html=True,
    )

    st.image('./src/assets/Bayes.png',width=250)


if __name__ == '__main__':
    main()

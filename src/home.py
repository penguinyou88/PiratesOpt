import streamlit as st


def main():
    st.markdown(
        '''
        <h1 align="center">
            Welcome to The <br/> 
            Pirates Treasure Hunt Game 👋
        </h1>

        ---

        #### About the Game

        A lost civilization hid a large number of diamonds of different sizes over some plots of land. You have been tasked by the King of your tribe to find the biggest diamond to help the tribe get enough supplies for the winter. However, a pirate tribe, captained by the famous pirate Thomas Bayes, is competing against you. Can you defeat the Bayes crew and find the bigger dimaond?
        
        * There are two levels corresponding to more difficult types of land. 
        * You and the Bayes crew are given prior information about diamond sizes in random locations. 
        * You only have a few guesses to try and improve upon the starting values. 
        * You only win if you get a higher value than the Bayes crew. 

        Click "Treasure Hunt Game" on the left to start playing. 

        ''',
        unsafe_allow_html=True,
    )

    st.image('./src/assets/Bayes.png',width=250)


if __name__ == '__main__':
    main()

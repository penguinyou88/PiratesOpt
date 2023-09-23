import streamlit as st

from src import home, about, mail, level1, level2

def init():
    # print("Running Initialization...")
    st.session_state.page = 'Homepage'
    st.session_state.project = False
    st.session_state.game = False

    st.session_state.pages = {
        'Homepage': home.main,
        'About Us': about.main,
        'Message Us': mail.main,
        'Level 1': level1.main,
        'Level 2': level2.main,
    }


def draw_style():
    st.set_page_config(page_title='Treasure Hunt Game', page_icon='📚')

    style = """
        <style>
            header {visibility: visible;}
            footer {visibility: hidden;}
        </style>
    """

    st.markdown(style, unsafe_allow_html=True)


def load_page():
    st.session_state.pages[st.session_state.page]()


def set_page(loc=None, reset=False):
    if not st.session_state.page == 'Homepage':
        for key in list(st.session_state.keys()):
            if key not in ('page', 'project', 'game', 'pages', 'set'):
                st.session_state.pop(key)

    if loc:
        st.session_state.page = loc
    else:
        st.session_state.page = st.session_state.set

    if reset:
        st.session_state.project = False
    elif st.session_state.page in ('Message Us', 'About Us'):
        st.session_state.project = True
        st.session_state.game = False
    else:
        pass


def change_button():
    set_page('Level 1')
    st.session_state.game = True
    st.session_state.project = True


def main():
    if 'page' not in st.session_state:
        init()

    draw_style()

    with st.sidebar:
        # project, about, contact = st.columns([1.1, 1, 1])

        if not st.session_state.project:
            st.sidebar.button('📌 Treasure Hunt Game', on_click=change_button)
        else:
            st.sidebar.button('🏠 Homepage', on_click=set_page, args=('Homepage', True))

        if st.session_state.project and st.session_state.game:
            st.selectbox(
                'Game Level',
                ['Level 1', 'Level 2'],
                key='set',
                on_change=set_page,
            )

        st.sidebar.button('🧑‍💻 About Us', on_click=set_page, args=('About Us',))

        st.sidebar.button('✉️ Message Us', on_click=set_page, args=('Message Us',))

        if st.session_state.page == 'Homepage':
            st.image('./src/assets/pirates.png')

    load_page()


if __name__ == '__main__':
    main()
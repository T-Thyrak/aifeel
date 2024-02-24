import streamlit as st

# Load the model
import dill as model_file


# model = model_file.load(open('model.dll', 'rb'))

def main():
    st.title('Predict Review Message Category')
    st.subheader('Enter the review message to predict the category')

    # review text with placeholder
    review = st.text_area('Review Message')
    result = ''
    if st.button('Predict'):
        st.subheader("Predicted Result:", divider="rainbow")
        # result = 1 if model.predict([review])[0] else 0
        result = 0
        positive_prob = 0.9
        negative_prob = 0.9

        # st.success(f'The review is {result}')
        # col2 is not used
        col1, col2 = st.columns(2)
        if result == 1:

            with col1:
                # percentage_positive = int(positive_prob * 100) show the title "Positive Predicted" , "Probability",
                # And slight light green color the background #ccffd2
                st.markdown(f'<div '
                            f'style="background-color: rgba(0, 255, 0, {positive_prob});'
                            f'padding: 10px; border-radius: 10px;">'
                            f'<h3>Positive Review</h3>'
                            f'<h4>Probability: {positive_prob}</h3>'
                            f'</div>'
                            f'', unsafe_allow_html=True)
            return

        with col1:
            # percentage_negative = int(negative_prob * 100)
            # show the title "Negative Predicted" , "Probability", And slight light red color the background #ffcccc
            st.markdown(f'<div style="background-color: rgba(200, 0, 0, {negative_prob});'
                        f' padding: 10px; border-radius: 10px;">'
                        f'<h3 style="color: white;">Negative Review</h3>'
                        f'<h4 style="color: white;">Probability: {negative_prob}</h3>'
                        f'</div>'
                        f'', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

# # ==========================================================================================
# # ======================== Scores 'With' Normalization ==============================
# # ==========================================================================================
#
# df_copy = df.copy()
# test_df_copy = test_df.copy()
#
# if transformation!='None':
#
#     df_copy_normal,test_df_copy_normal,normal_object = backend.data_transformation(df_copy,test_df_copy,transformation)
#
#     train_data_normal = df_copy_normal.set_index('date')
#     train_data_normal = train_data_normal['sold']
#     # train_data_normal = train_data_normal.asfreq('D')
#
#     test_data_normal = test_df_copy_normal.set_index('Date')
#     test_data_normal = test_data_normal['number_sold']
#     # test_data = test_data.asfreq('D')
#
#     arima_normal = ARIMA(train_data_normal, order=(p, d, q))
#     model_normal = arima_normal.fit()
#     predicted_normal = model_normal.forecast(steps=forecast_size)
#     mse_score_normal = mean_squared_error(test_data_normal[0:forecast_size], predicted_normal)
#     rmse_score_normal = np.sqrt(mean_squared_error(test_data_normal[0:forecast_size], predicted_normal))
#
#
#     # ------------------------ Inverse transformation -------------------------------
#
#     test_data_normal_inverse = None
#     predicted_normal_inverse = None
#
#     if normal_object is not None:
#         predicted_normal_inverse = normal_object.inverse_transform(predicted_normal.values.reshape(-1,1))
#         test_data_normal_inverse = normal_object.inverse_transform(test_data_normal.values.reshape(-1,1))
#     elif transformation == 'Square Root Transform':
#         predicted_normal_inverse = np.square(predicted_normal)
#         test_data_normal_inverse = np.square(test_data_normal)
#     elif transformation == 'Log Transform':
#         predicted_normal_inverse = np.exp(predicted_normal)
#         test_data_normal_inverse = np.exp(test_data_normal)
#
#     mse_score_normal_inverse = mean_squared_error(test_data_normal_inverse[0:forecast_size],predicted_normal_inverse)
#     rmse_score_normal_inverse = np.sqrt(mean_squared_error(test_data_normal_inverse[0:forecast_size], predicted_normal_inverse))
#
#     # ======================== Printing the SCORES ==============================
#
#     col_10_no, col_11_no, col_12_no, col_13_no = st.columns(4)
#     with col_10_no:
#         st.markdown("<h4 style='text-align: center;'>No Transformation:</h4>", unsafe_allow_html=True)
#     with col_11_no:
#         st.markdown("<h4 style='text-align: center;'>📌 MSE Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(mse_score, 2)}</h4>", unsafe_allow_html=True)
#     with col_12_no:
#         st.markdown("<h4 style='text-align: center;'>📌 RMSE Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(rmse_score, 2)}</h4>", unsafe_allow_html=True)
#     with col_13_no:
#         st.markdown("<h4 style='text-align: center;'>📌 AIC Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(model.aic, 2)}</h4>", unsafe_allow_html=True)
#
#
#     st.markdown('---')
#
#     col100, col_111, col_122, col_133 = st.columns(4)
#     with col100:
#         st.markdown("<h4 style='text-align: center;'>After Transformation:</h4>", unsafe_allow_html=True)
#     with col_111:
#         st.markdown("<h4 style='text-align: center;'>📌 MSE Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(mse_score_normal, 2)}</h4>", unsafe_allow_html=True)
#     with col_122:
#         st.markdown("<h4 style='text-align: center;'>📌 RMSE Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(rmse_score_normal, 2)}</h4>",
#                     unsafe_allow_html=True)
#     with col_133:
#         st.markdown("<h4 style='text-align: center;'>📌 AIC Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(model_normal.aic, 2)}</h4>", unsafe_allow_html=True)
#
#     st.markdown('---')
#
#     col_inv1, col_inv2, col_inv3, col_inv4 = st.columns(4)
#     with col_inv1:
#         st.markdown(f"<h4 style='text-align: center;'>After Inverse {transformation}:</h4>", unsafe_allow_html=True)
#     with col_inv2:
#         st.markdown("<h4 style='text-align: center;'>📌 MSE Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(mse_score_normal_inverse, 2)}</h4>",
#                     unsafe_allow_html=True)
#     with col_inv3:
#         st.markdown("<h4 style='text-align: center;'>📌 RMSE Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(rmse_score_normal_inverse, 2)}</h4>",
#                     unsafe_allow_html=True)
#     with col_inv4:
#         st.markdown("<h4 style='text-align: center;'>📌 AIC Score:</h4>", unsafe_allow_html=True)
#         st.markdown(f"<h4 style='text-align: center;'>{np.round(model_normal.aic, 2)}</h4>", unsafe_allow_html=True)
#
#
#     # ------------------------Final result graph ----------------------------------------------------
#
#     #
#     # st.text(predicted)
#     # st.text(forecast_size)
#
#
#     # ==========================================================================================
#     # ======================== Scores Without Any kind of Normalization - 3 columns ============
#     # ==========================================================================================
# else:
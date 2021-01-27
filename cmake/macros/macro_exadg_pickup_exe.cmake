MACRO(EXADG_PICKUP_EXE FILE_NAME TARGET_NAME EXE_NAME)

    ADD_EXECUTABLE(${TARGET_NAME} ${FILE_NAME})
    DEAL_II_SETUP_TARGET(${TARGET_NAME})
    SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${EXE_NAME})
    TARGET_LINK_LIBRARIES(${TARGET_NAME} exadg)

    IF(${USE_FFTW})
       TARGET_LINK_LIBRARIES(${TARGET_NAME} ${FFTW3_MPI})
       TARGET_LINK_LIBRARIES(${TARGET_NAME} ${FFTW3})
    ENDIF()

ENDMACRO(EXADG_PICKUP_EXE)

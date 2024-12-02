import math

def format_prompt(text, emotion, temperature_adjuster=0.5):

    temperature_adjuster = 2 * temperature_adjuster
    

    temperature = 0.01
    top_p = 0.9
    repetition_penalty = 1.05
    custom_rep_pen = 1.0
    top_k = 0

    if emotion == "crying":
        temperature = 0.8
        top_p = 0.95
        repetition_penalty = 1.05
        custom_rep_pen=1.05

    if emotion == "sad":
        temperature = 0.7
        top_p = 0.95
        repetition_penalty = 1.05
        custom_rep_pen=1.2

    if  emotion == "angry":
        temperature = 0.7
        top_p = 0.95
        repetition_penalty = 1.05
        custom_rep_pen=1.2


    #     model=model,
    # input_ids=input_ids,
    # max_length=2000,
    # temperature = 0.7,
    # top_p = 0.95,
    # repetition_penalty = 1.05,
    # custom_rep_pen=1.2


    if emotion == "normal":
        emotion = "happy" 


    if  emotion == "happy":
        temperature = 0.5
        top_p = 0.8
        repetition_penalty = 1.1
        custom_rep_pen=1.05
        top_k = 50




    if emotion == "curious":
        temperature = 0.7
        top_p = 0.8
        repetition_penalty = 1.05
        custom_rep_pen=1.1



    if emotion == "whisper":
        temperature = 0.95
        top_p = 0.95
        repetition_penalty = 1.05
        custom_rep_pen=1.0

    if emotion == "slow":
        temperature = 0.5
        top_p = 0.8
        repetition_penalty = 1.05
        custom_rep_pen=1.05


    if emotion == "disgust":
        temperature = 0.5
        top_p = 0.98
        repetition_penalty = 1.05
        custom_rep_pen=1.05

    temperature = temperature_adjuster * temperature
    custom_rep_pen = custom_rep_pen / math.sqrt(temperature_adjuster)
    prompt = f"<{emotion}> {text} </{emotion}>"

    return prompt, temperature, top_p, repetition_penalty, custom_rep_pen, top_k
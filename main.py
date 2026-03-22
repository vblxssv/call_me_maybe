from llm_sdk import Small_LLM_Model



def main():
    model = Small_LLM_Model()
    text = """залупа пенис хер"""

    tensor = model.encode(text)
    print(tensor)

    try_decode = model.decode(tensor)
    print(try_decode)

    try_decode2 = model.decode(tensor[0].tolist())
    print(try_decode2)




if __name__ == "__main__":
    main()

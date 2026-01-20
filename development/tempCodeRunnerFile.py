

    # If nothing was provided on the CLI, ask interactively instead of erroring.
    if not user_text:
        user_text = input("Enter a short text to describe the color: ").strip()

    if not user_text:
        parser.error("Please provide a text prompt, e.g. python predict_color.py \"calm ocean\"")

    vectorizer, model = load_artifacts()
    rgb = predict_rgb(user_text, vectorizer, model)
    hex_code = to_hex(rgb)

    result = {"input": user_text, "rgb": rgb, "hex": hex_code}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
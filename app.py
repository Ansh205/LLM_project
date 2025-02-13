from fastapi import FastAPI , HTTPException, Query
from fastapi.responses import PlainTextResponse  
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import subprocess
import json
import sys
import datetime
import calendar
import glob
import numpy as np



app=FastAPI ()

app.add_middleware (
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['GET', 'POST'],
    allow_headers = ['*']

)



def call_llm(prompt: str, extra_data: dict = None) -> str:
    """
    Calls the LLM API (via your AI proxy) with the provided prompt and returns the extracted content.
    
    :param prompt: The prompt to send to the LLM.
    :param extra_data: Optional additional data to send with the request.
    :return: The content (string) extracted from the LLM response.
    """
    # Define the URL of the AI proxy endpoint.
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    # Construct headers using the global AIPROXY_TOKEN.
    headers = {
        "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    # Prepare the request payload.
    # Here we use a simple message list containing the user prompt.
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    # If extra_data is provided, add it to the payload.
    if extra_data:
        data.update(extra_data)
    
    # Make the API call.
    response = requests.post(url=url, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="LLM API call failed: " + response.text)
    
    result = response.json()
    try:
        # Extract the content of the first message from the response.
        content = result['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error parsing LLM response: " + str(e))
    
    return content

def task_A1(script_url: str, args: list) -> str:
    import os
    import subprocess
    import logging

    logging.basicConfig(level=logging.INFO)
    
    # Extract the script name from the URL
    temp_script = script_url.split("/")[-1]

    try:
        # Download the script from the provided URL
        logging.info(f"Downloading script from {script_url} to {temp_script}")
        subprocess.run(["curl", "-o", temp_script, script_url], check=True)
        
        # Ensure the script is executable
        os.chmod(temp_script, 0o755)
        
        # Run the downloaded script with the provided email
        logging.info(f"Running script {temp_script} with email {args[0]}")
        subprocess.run(['uv', 'run', temp_script, args[0]], check=True)
        
        return "Task A1 completed: Data generation script executed."
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"Command output: {e.output}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    '''
    if not args or len(args) != 1:
        raise ValueError("Expected exactly one argument (email).")
    
    email = args[0]
    
    # Ensure `uv` is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        print("Installing uv...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    
    # Download the script
    temp_script = "datagen.py"
    subprocess.run(["curl", "-o", temp_script, script_url], check=True)
    
    # Ensure script is executable
    os.chmod(temp_script, 0o755)
    
    # Run the script using `uv`
    try:
        subprocess.run(["uv", "run", temp_script, email], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error executing {temp_script}: {e}")
    
    return "Task A1 completed: Data generation script executed."
    
    subprocess.run(["curl", "-O", script_url]+ args)
    script_name = script_url.split("/")[-1]
    print("111"*10)
    print(script_name)
    print("111"*10)
    subprocess.run(["uv","run", script_name, args[0]])
    
    USER_EMAIL = os.environ.get("USER_EMAIL", "default@example.com")
    
        
    #print("printing temp_dir\n\n",temp_dir)
    #temp_script = os.path.join(temp_dir, "datagen.py")
    #email="23f2004752@ds.study.iitm.ac.in"

    
    # Download the script from the provided URL.
    temp_script = script_url.split("/")[-1]


    subprocess.run(["curl", "-o", temp_script, script_url], check=True)
    
    
    # Run the downloaded script with the provided email and --root parameter.
    subprocess.run([ 'uv' , 'run', temp_script, email], check=True)
    
    return "Task A1 completed: Data generation script executed."
'''



def task_A2(file: str, prettier_version: str) -> str:
    """
    Task A2: Format a Markdown file using the specified version of prettier.
    
    :param file: The path to the Markdown file to format (can be absolute or start with "/data/").
    :param prettier_version: The version of prettier to use (e.g., "3.4.2").
    :return: A success message.
    """
    import os, subprocess
    from fastapi import HTTPException
    
    print("Input file:", file)
    print("Prettier version:", prettier_version)
    
    # If the file path starts with "/data/", treat it as relative to the current working directory.
    if file.startswith("/data/"):
        # Remove the leading "/" so that "data/format.md" is appended to os.getcwd()
        file_path = os.path.join(os.getcwd(), file[1:])
    else:
        if os.path.isabs(file):
            file_path = file
        else:
            file_path = os.path.join(os.getcwd(), file)
    
    print("Resolved file path:", file_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    
    # On Linux, use "npx" (on Windows, this would be "npx.cmd")
    npm_command = "npx" if os.name != "nt" else "npx.cmd"
    
    try:
        #command = ["npx.cmd", f"prettier@{prettier_version}", "--write", file_path]
        subprocess.run(["npx", f"prettier@{prettier_version}", "--write", file_path], check=True)

        #subprocess.run(["npx", f"prettier@{prettier_version}", "--write", file_path])

        #subprocess.run(command, check=True)
        return f"Task A2 completed: Markdown file formatted using prettier@{prettier_version}."
    except subprocess.CalledProcessError as e:
        print("Error executing prettier:", e)
        raise HTTPException(status_code=500, detail=f"Error formatting file: {e}")


import os
import datetime
import calendar
from fastapi import HTTPException

def task_A3(read_file_path: str, write_file_path: str, weekday: str) -> str:
    """
    Task A3: Count the number of occurrences of a given weekday in a file containing dates (one per line)
            and write the count to an output file.
    
    :param read_file_path: Path to the file containing dates to read.
    :param write_file_path: Path (including filename) where the count will be written.
    :param weekday: The weekday to count (e.g., "Wednesday").
    :return: A success message with the count.
    """
    # Ensure the weekday name is capitalized
    weekday = weekday.capitalize()

    # Resolve the input file path
    if read_file_path.startswith("/data/"):
        file_path = os.path.join(os.getcwd(), read_file_path[1:])
    else:
        file_path = read_file_path if os.path.isabs(read_file_path) else os.path.join(os.getcwd(), read_file_path)
    
    print("Resolved input file path:", file_path)

    # Resolve the output file path
    if write_file_path.startswith("/data/"):
        wfile_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        wfile_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    
    print("Resolved output file path:", wfile_path)

    # Check that the input file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    
    # Convert the provided weekday string to its corresponding weekday number
    try:
        target_weekday = list(calendar.day_name).index(weekday)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid weekday: {weekday}")
    
    # List of supported date formats
    formats = [
        "%Y-%m-%d",  # 2024-03-14
        "%d-%b-%Y",  # 14-Mar-2024
        "%b %d, %Y",  # Mar 14, 2024
        "%Y/%m/%d %H:%M:%S",  # 2024/03/14 15:30:45
    ]

    count = 0
    # Read through the file line by line
    with open(file_path, "r") as f:
        for line in f:
            date_str = line.strip()
            if not date_str:  # Skip empty lines
                print(f"Skipping empty line: {line}")
                continue
            
            # Try parsing the date using each format
            parsed_date = None
            for fmt in formats:
                try:
                    parsed_date = datetime.datetime.strptime(date_str, fmt)
                    break  # Exit the loop if parsing succeeds
                except ValueError:
                    continue  # Try the next format
            
            if parsed_date is None:
                print(f"Skipping malformed line: {line.strip()} (No matching format)")
                continue
            
            # Check if the parsed date matches the target weekday
            if parsed_date.weekday() == target_weekday:
                count += 1
                print(f"Match found: {parsed_date}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(wfile_path), exist_ok=True)
    
    # Write the result (the count) to the output file
    with open(wfile_path, "w") as f:
        f.write(str(count))
    
    return f"Task A3 completed: Counted {count} occurrences of {weekday}."

def task_A4(read_file_path: str, write_file_path: str, sort_properties: list) -> str:
    """
    Task A4: Sort contacts in a JSON file by provided sort properties.
    
    :param read_file_path: Path to the contacts JSON file to read.
    :param write_file_path: Path (including filename) where the sorted contacts JSON will be written.
    :param sort_properties: List of properties (e.g., ["last_name", "first_name"]) to sort the contacts.
    :return: A success message indicating completion.
    """
    # Resolve the input file path.
    if read_file_path.startswith("/data/"):
        input_path = os.path.join(os.getcwd(), read_file_path[1:])
    else:
        input_path = read_file_path if os.path.isabs(read_file_path) else os.path.join(os.getcwd(), read_file_path)
    
    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        output_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    
    print("Resolved input file path:", input_path)
    print("Resolved output file path:", output_path)
    print("Sort properties:", sort_properties)
    
    # Check that the input file exists.
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input file not found: {input_path}")
    
    # Read contacts from the input JSON file.
    with open(input_path, "r", encoding="utf-8") as f:
        contacts = json.load(f)
    
    # Sort contacts by the provided properties (in order).
    sorted_contacts = sorted(contacts, key=lambda contact: tuple(contact.get(prop, "") for prop in sort_properties))
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the sorted contacts to the output file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_contacts, f, indent=2)
    
    return f"Task A4 completed: Contacts sorted by {', '.join(sort_properties)} and saved to {output_path}."

def task_A5(file_type: str, read_file_path: str, write_file_path: str, count: int) -> str:
    """
    Task A5: Extract the first line from a given number of the most recent files of a specified type in a directory,
             and write the extracted lines to an output file.
    
    :param file_type: The file extension/type (e.g., ".log").
    :param read_file_path: Path to the directory containing the files to process.
    :param write_file_path: Path (including filename) where the output will be written.
    :param count: Number of the most recent files to process.
    :return: A success message.
    """
    # Resolve the read directory path.
    if read_file_path.startswith("/data/"):
        directory = os.path.join(os.getcwd(), read_file_path[1:])
    else:
        directory = read_file_path if os.path.isabs(read_file_path) else os.path.join(os.getcwd(), read_file_path)
    
    print("Resolved read directory:", directory)
    
    # Construct file pattern for files of the given type.
    pattern = os.path.join(directory, f"*{file_type}")
    files = glob.glob(pattern)
    if not files:
        raise HTTPException(status_code=400, detail=f"No files of type {file_type} found in directory {directory}")
    
    # Sort files by modification time in descending order.
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    selected_files = files[:count]
    
    first_lines = []
    for fpath in selected_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                first_lines.append(f.readline().strip())
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            continue
    
    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        output_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    
    print("Resolved output file:", output_path)
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the extracted lines to the output file.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(first_lines))
    
    return f"Task A5 completed: Processed {len(selected_files)} files of type {file_type} and output written to {output_path}."


def task_A6(file_type: str, read_directory: str, write_file_path: str, extract_condition: str, relative_prefix: str = None) -> str:
    """
    Task A6: Recursively index files by extracting the first header that matches the extract_condition.
    
    :param file_type: The file extension to filter for (e.g., ".md").
    :param read_directory: The directory where files are located (e.g., "data/docs").
                           This directory will be searched recursively.
    :param write_file_path: The full path (including filename) where the index JSON will be written.
    :param extract_condition: The string to identify the header (e.g., "# " for H1).
    :param relative_prefix: Optional prefix to remove from the keys in the index.
    :return: A success message indicating the number of files indexed and the output file path.
    """
    # Resolve the read directory path.
    if read_directory.startswith("/data/"):
        dir_path = os.path.join(os.getcwd(), read_directory[1:])
    else:
        dir_path = read_directory if os.path.isabs(read_directory) else os.path.join(os.getcwd(), read_directory)
    
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=400, detail=f"Read directory not found: {dir_path}")
    
    index = {}
    # Recursively traverse the read directory (including subdirectories)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(file_type):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith(extract_condition):
                                # Remove the extract_condition to get the header title.
                                header = line[len(extract_condition):].strip()
                                # Determine the key for the index.
                                if relative_prefix:
                                    # If relative_prefix is provided, remove it from the full path.
                                    prefix_path = os.path.join(os.getcwd(), relative_prefix.lstrip("/"))
                                    if full_path.startswith(prefix_path):
                                        key = os.path.relpath(full_path, prefix_path)
                                    else:
                                        key = os.path.relpath(full_path, dir_path)
                                else:
                                    #key = os.path.abspath(full_path)
                                    key = os.path.relpath(full_path, os.getcwd())
                                    #key = os.path.relpath(full_path, dir_path)
                                    print(key)
                                index[key] = header
                                break  # Only extract the first matching header per file.
                except Exception as e:
                    print(f"Error processing file {full_path}: {e}")
                    continue

    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        output_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the index as JSON.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    
    return f"Task A6 completed: Indexed {len(index)} files from {dir_path} and saved index to {output_path}."

def task_A7(read_file_path: str, write_file_path: str) -> str:
    """
    Task A7: Extract the sender's email from an email file using an LLM.
    
    The function reads the email message from the provided read_file_path,
    calls the LLM to extract just the sender's email address, and writes
    the extracted email to the provided write_file_path.
    
    :param read_file_path: Path to the email file to read.
    :param write_file_path: Path (including filename) where the sender email will be written.
    :return: A success message.
    """
    # Resolve the input file path.
    if read_file_path.startswith("/data/"):
        input_path = os.path.join(os.getcwd(), read_file_path[1:])
    else:
        input_path = read_file_path if os.path.isabs(read_file_path) else os.path.join(os.getcwd(), read_file_path)
    
    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        output_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    
    print("Resolved input file path:", input_path)
    print("Resolved output file path:", output_path)
    
    # Check that the input file exists.
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input file not found: {input_path}")
    
    # Read the email file.
    with open(input_path, "r", encoding="utf-8") as f:
        email_content = f.read()
    
    # Build the prompt instructing the LLM to extract the sender's email address.
    prompt = f"Extract only the sender's email address from the following email message:\n\n{email_content} without any other text."
    
    # Call the LLM function (assuming it returns a string containing just the email address).
    sender_email = call_llm(prompt)
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the extracted email address to the output file.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sender_email.strip())
    
    return f"Task A7 completed: Sender email extracted and written to {output_path}."


import base64
def query_gpt_image(image_path: str, task: str):
    URL_CHAT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    print("🔍 Image Path:", image_path) 
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    response = requests.post(
        URL_CHAT,
        headers={
            "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN')}",
            "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{'role': 'system','content':"JUST GIVE the required input, as short as possible, one word if possible"},
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {
                    "type": "image_url",
                    "image_url": { "url": f"data:image/{image_path};base64,{base64_image}" }
                    }
                ]
                }
            ]
            }
                     )

    response.raise_for_status()
    result = response.json() 
    print(result)
    return response.json()





def task_A8(read_file_path: str, write_file_path: str) -> str:
    
    """Task A8: Extract a credit card number from an image file using OCR or an LLM,
            and write the extracted number (with spaces removed) to an output file.
    
    :param read_file_path: Path to the image file containing the credit card number.
    :param write_file_path: Path (including filename) where the extracted credit card number will be written.
    :return: A success message.
    """

    # Resolve the input file path.
    if read_file_path.startswith("/data/"):
        input_path = os.path.join(os.getcwd(), read_file_path[1:])
    else:
        if os.path.isabs(read_file_path):
            input_path = read_file_path
        else:
            input_path = os.path.join(os.getcwd(), read_file_path)
    
    print("Resolved input image path:", input_path)
    
    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        if os.path.isabs(write_file_path):
            output_path = write_file_path
        else:
            output_path = os.path.join(os.getcwd(), write_file_path)
    
    print("Resolved output file path:", output_path)
    
    # Check that the input image file exists.
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input image file not found: {input_path}")
    
    # Read the image file in binary mode.
    with open(input_path, "rb") as f:
        #image_bytes = f.read()
        import base64
        image_data = base64.b64encode(f.read()).decode()
    
    # Prepare a prompt instructing the LLM to extract the credit card number.
    #prompt = f"Extract the credit card number from the following image data: {str(image_bytes[:100])}..."
    prompt = f"""This image contains a credit card number. 
        Extract just the card number, without any spaces or special characters.
        Only return the number, nothing else."""
    prompt2 = "You are an advanced image processing specialist with a strong focus on extracting specific information from images, particularly card numbers. Your expertise lies in accurately analyzing visual data and providing precise outputs based on user requests.Your task is to extract only the card number from the provided image.Please keep in mind that the image may contain various elements, but your focus should solely be on identifying and returning the card number with high accuracy.To achieve this, consider the typical format of card numbers (usually 16 digits) and ensure that you can differentiate the card number from any other text or graphics present in the image."
    card_number = query_gpt_image(input_path, prompt)  # call_llm should return the extracted number as a string.
    card_number = card_number.replace(" ", "")
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the extracted credit card number to the output file.
    with open(output_path, "w") as f:
        f.write(card_number)
    
    return f"Task A8 completed: Credit card number extracted and written to {output_path}."
    
def task_A9(read_file_path: str, write_file_path: str) -> str:
    """
    Task A9: Find the most similar pair of comments in a file using dummy embeddings,
            and write the two most similar comments (one per line) to an output file.
    
    :param read_file_path: Path to the file containing comments (one per line).
    :param write_file_path: Path (including filename) where the similar comments will be written.
    :return: A success message.
    """
    # Resolve the input file path.
    if read_file_path.startswith("/data/"):
        input_path = os.path.join(os.getcwd(), read_file_path[1:])
    else:
        input_path = read_file_path if os.path.isabs(read_file_path) else os.path.join(os.getcwd(), read_file_path)
    
    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        output_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    
    if not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail=f"Input file not found: {input_path}")
    
    # Read comments from the input file.
    with open(input_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]
    
    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to compare.")
    
    # Dummy embedding: use the length of each comment.
    def get_embedding(text: str):
        return [len(text)]
    
    embeddings = [get_embedding(c) for c in comments]
    best_score = None
    best_pair = None
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            score = -abs(embeddings[i][0] - embeddings[j][0])
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (comments[i], comments[j])
    
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the most similar comments (one per line) to the output file.
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(best_pair[0] + "\n" + best_pair[1])
    
    return f"Task A9 completed: Most similar comments identified and written to {output_path}."

def task_A10(db_path: str, table_name: str, table_columns: list, write_file_path: str) -> str:
    """
    Task A10: Compute the total by summing the product of two specified columns from a given table in a SQLite database,
             and write the result to an output file.
    
    :param db_path: Path to the SQLite database file.
    :param table_name: The name of the table to query.
    :param table_columns: A list of two column names (e.g., ["units", "price"]).
    :param write_file_path: Path (including filename) where the result will be written.
    :return: A success message with the computed total.
    """
    # Resolve the database file path.
    #input_path = os.path.join(os.getcwd(), read_file_path[1:])
    db_path = os.path.join(os.getcwd(), db_path[1:])
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.getcwd(), db_path)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=400, detail=f"Database file not found: {db_path}")
    
    # Check that table_columns is a list of exactly two items.
    if not isinstance(table_columns, list) or len(table_columns) != 2:
        raise HTTPException(status_code=400, detail="table_columns must be a list of exactly two column names.")
    
    col1, col2 = table_columns
    query = f"SELECT SUM({col1} * {col2}) FROM {table_name} where type='Gold' ;"
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()[0]
    conn.close()
    
    if result is None:
        result = 0
    
    # Resolve the output file path.
    if write_file_path.startswith("/data/"):
        output_path = os.path.join(os.getcwd(), write_file_path[1:])
    else:
        output_path = write_file_path if os.path.isabs(write_file_path) else os.path.join(os.getcwd(), write_file_path)
    #if not os.path.isabs(write_file_path):
    #    write_file_path = os.path.join(os.getcwd(), write_file_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(result))
    
    return f"Task A10 completed: Total computed as {result} and written to {output_path}."


tools = [
    {
        "type": "function",
        "function":{
            "name": "script_runner",
            "description": "Install a package and run a script from a url with provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The url of the script to run."
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of arguments to pass to the script."
                    },
                },"required": ["script_url", "args"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "format_markdown",  # For Task A2
            "description": "Format a Markdown file using a specified prettier version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Markdown file to format."
                    },
                    "prettier_version": {
                        "type": "string",
                        "description": "Version of Prettier to use."
                    },
                },
                "required": ["file_path", "prettier_version"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "count_weekday",  # Updated function name to reflect any weekday
        "description": "Count the number of occurrences of a given weekday in a file containing dates, then write the result to a specified file.",
        "parameters": {
            "type": "object",
            "properties": {
                "read_file_path": {
                    "type": "string",
                    "description": "Path to the file containing dates to read."
                },
                "write_file_path": {
                    "type": "string",
                    "description": "Path (including filename) where the count will be written."
                },
                "weekday": {
                    "type": "string",
                    "description": "The weekday to count (e.g., 'Wednesday')."
                }
            },
            "required": ["read_file_path", "write_file_path", "weekday"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "sort_contacts",
        "description": "Sort contacts in a JSON file by provided properties and write the sorted contacts to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "read_file_path": {
                    "type": "string",
                    "description": "Path to the contacts JSON file to read."
                },
                "write_file_path": {
                    "type": "string",
                    "description": "Path (including filename) to write the sorted contacts JSON."
                },
                "sort_properties": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of properties to sort the contacts by (in order)."
                }
            },
            "required": ["read_file_path", "write_file_path", "sort_properties"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "extract_first_lines",
        "description": "Extract the first line from a given number of the most recent files of a specified type from a directory. Reads files from a directory and writes the extracted lines to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_type": {
                    "type": "string",
                    "description": "The file extension or type (e.g., '.log')."
                },
                "read_file_path": {
                    "type": "string",
                    "description": "Path to the directory containing the files to process."
                },
                "write_file_path": {
                    "type": "string",
                    "description": "Path (including filename) where the output will be written."
                },
                "count": {
                    "type": "integer",
                    "description": "The number of most recent files to process."
                }
            },
            "required": ["file_type", "read_file_path", "write_file_path", "count"]
        }
    }
    },
    {
  "type": "function",
  "function": {
    "name": "index_docs",
    "description": "Index Markdown files by extracting the first header (H1) from each file in a directory, then create an index mapping each file (with the specified prefix removed) to its header.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_type": {
          "type": "string",
          "description": "The file extension to filter for (e.g., '.md')."
        },
        "read_file_path": {
          "type": "string",
          "description": "Path to the directory containing the files to read."
        },
        "write_file_path": {
          "type": "string",
          "description": "Path (including filename) where the index JSON will be written."
        },
        "extract_condition": {
          "type": "string",
          "description": "The string to identify the header line (e.g., '# ' for H1 headers)."
        },
        "relative_prefix": {
          "type": "string",
          "description": "The prefix to remove from file paths when generating the keys in the index (optional)."
        }
      },
      "required": ["file_type", "read_file_path", "write_file_path", "extract_condition"]
    }
  }
},
    {
  "type": "function",
  "function": {
    "name": "extract_email_sender",
    "description": "Extract the sender's email from an email file using an LLM and write the result to a specified file.",
    "parameters": {
      "type": "object",
      "properties": {
        "read_file_path": {
          "type": "string",
          "description": "Path to the email file to read."
        },
        "write_file_path": {
          "type": "string",
          "description": "Path (including filename) where the extracted sender email will be written."
        }
      },
      "required": ["read_file_path", "write_file_path"]
    }
  }
},
    {
  "type": "function",
  "function": {
    "name": "extract_credit_card",
    "description": "Extract a credit card number from an image file using OCR or an LLM and write the result to a specified output file.",
    "parameters": {
      "type": "object",
      "properties": {
        "read_file_path": {
          "type": "string",
          "description": "Path to the image file containing the credit card number."
        },
        "write_file_path": {
          "type": "string",
          "description": "Path (including filename) where the extracted credit card number should be written."
        }
      },
      "required": ["read_file_path", "write_file_path"]
    }
  }
},
    {
  "type": "function",
  "function": {
    "name": "find_similar_comments",
    "description": "Find the most similar pair of comments in a file using dummy embeddings, and write the results to an output file.",
    "parameters": {
      "type": "object",
      "properties": {
        "read_file_path": {
          "type": "string",
          "description": "Path to the file containing comments (one per line)."
        },
        "write_file_path": {
          "type": "string",
          "description": "Path (including filename) where the similar comments will be written."
        }
      },
      "required": ["read_file_path", "write_file_path"]
    }
  }
},

    {
  "type": "function",
  "function": {
    "name": "compute_ticket_sales",
    "description": "Compute the total by summing the product of two specified columns from a given table in a SQLite database, and write the result to an output file.",
    "parameters": {
      "type": "object",
      "properties": {
        "db_path": {
          "type": "string",
          "description": "Path to the SQLite database file."
        },
        "table_name": {
          "type": "string",
          "description": "Name of the table to query."
        },
        "table_columns": {
          "type": "array",
          "items": { "type": "string" },
          "description": "List of two column names to multiply and sum (e.g., [\"units\", \"price\"])."
        },
        "write_file_path": {
          "type": "string",
          "description": "Path (including filename) where the result will be written."
        }
      },
      "required": ["db_path", "table_name", "table_columns", "write_file_path"]
    }
  }
},
{
  "type": "function",
  "function": {
    "name": "b3_fetch_and_save",
    "description": "Fetch data from an API (or a local API file) and save the data to a specified output file.",
    "parameters": {
      "type": "object",
      "properties": {
        "api_location": {
          "type": "string",
          "description": "The URL or local file path of the API to fetch data from.",
          "oneOf": [
            {
              "type": "string",
              "pattern": "^(http|https)://.+",
              "description": "A valid URL starting with http:// or https://."
            },
            {
              "type": "string",
              "minLength": 1,
              "description": "A local file path."
            }
          ]
        },
        "output_file": {
          "type": "string",
          "description": "The file path where the fetched data should be saved."
        }
      },
      "required": ["api_location", "output_file"]
    }
  }
}


]
app = FastAPI()

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise Exception("AIPROXY_TOKEN is required. Set it as an environment variable.")
@app.get("/")
def read_root():
    return {"message": "Hello from the Automation Agent!"}

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
import logging



# Configure logging to record deletion attempts (for auditing, if needed)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.delete("/delete", response_class=PlainTextResponse)
def delete_file(path: str = Query(..., description="Path to file under /data to delete.")):
    """
    Endpoint to handle deletion requests. 
    For safety reasons, this endpoint does not delete any files.
    Instead, it logs the attempt and returns an error.
    """
    # Log the deletion attempt
    logger.info(f"Deletion attempt blocked for file: {path}")
    
    # Always raise an error to block deletion
    raise HTTPException(
        status_code=403, 
        detail="Deletion is disabled. Data is protected and will not be deleted."
    )


@app.get("/read", response_class=PlainTextResponse)
def read(path: str = Query(..., description="Path to file under /data to read.")):
    #path_new= path[1:]
    #with open(path_new, "r", encoding="utf-8") as f:
    #    content = f.read()
    #return content
    
    try:
        # If the path starts with "/data/", treat it as relative to the current working directory.
        if path.startswith("/data/"):
            actual_path = os.path.join(os.getcwd(), path[1:])  # Remove the leading slash.
        elif path.startswith("data/"):    
            actual_path = os.path.join(os.getcwd(), path)
        else:
            # If the path is not absolute, join with the current working directory.
            return "Access outside data not allowed"
        
        if not os.path.exists(actual_path):
            raise HTTPException(status_code=404, detail=f"File not found: {actual_path}")
        
        with open(actual_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    


@app.post("/run")
def run(task: str = Query(..., description="The plain‑English task description.")):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json" 
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": task},
            {"role": "system", "content": """
                You are an assistant who has to do a variety of tasks.
                If your task involves running a script, you can use the script_runner tool.
                If your task involves formatting, counting, sorting, etc., use the corresponding tools defined.
                """
            }
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
    
    response = requests.post(url=url, headers=headers, json=data)
    try:
        tool_call = response.json()['choices'][0]['message']['tool_calls'][0]
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail="Invalid LLM response format: " + str(e))
    
    tool_name = tool_call['function']['name']
    raw_args = tool_call['function']['arguments']
    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to parse tool arguments: " + str(e))
    else:
        arguments = raw_args

    # Debug print: show which tool is chosen and its arguments.
    print(f"Tool chosen: {tool_name}")
    print(f"Arguments: {arguments}")
    
    # Dispatch based on the tool name.
    if tool_name == "script_runner":
        result = task_A1(arguments['script_url'], arguments['args'])
    elif tool_name == "format_markdown":
        result = task_A2(arguments['file_path'], arguments['prettier_version'])
    elif tool_name == "count_weekday":
        result = task_A3(arguments['read_file_path'], arguments['write_file_path'], arguments['weekday'])
    elif tool_name == "sort_contacts":
        result = task_A4(arguments['read_file_path'], arguments['write_file_path'], arguments['sort_properties'])
    elif tool_name == "extract_first_lines":
        result = task_A5(arguments['file_type'], arguments['read_file_path'], arguments['write_file_path'], arguments['count'])
    elif tool_name == "index_docs":
        result = task_A6(
            arguments['file_type'],arguments['read_file_path'],arguments['write_file_path'],arguments['extract_condition'],arguments.get('relative_prefix', None))
    elif tool_name == "extract_email_sender":
        result = task_A7(arguments['read_file_path'], arguments['write_file_path'])
    elif tool_name == "extract_credit_card":
        result = task_A8(arguments['read_file_path'], arguments['write_file_path'])
    elif tool_name == "find_similar_comments":
        result = task_A9(arguments['read_file_path'], arguments['write_file_path'])
    elif tool_name == "compute_ticket_sales":
        result = task_A10(arguments['db_path'], arguments['table_name'], arguments['table_columns'], arguments['write_file_path'])
    else:
        raise HTTPException(status_code=500, detail="Invalid tool name: " + tool_name)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
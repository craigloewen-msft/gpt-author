import ast
import json
import random
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
import openai
import os
from ebooklib import epub
import base64
import os
import requests

from io import BytesIO

from PIL import Image

openai.api_type = "azure"
openai.api_base = "https://azureopenaitesting.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "f83b18ce42c245ef98f0581683918677"

engineName = "gptauthor"
chosenLanguage = "beginner Korean"


# get it at https://beta.dreamstudio.ai/
stability_api_key = "ENTER STABILITY KEY HERE"

def check_and_load_cache(inputName):
    file_path = "./content/" + inputName + ".txt"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    return None

def save_content(inputName, content):
    file_path = "./content/" + inputName + ".txt"
    with open(file_path, 'w') as file:
        file.write(content)

def generate_cover_prompt(plot):
    cacheName = "cover_prompt"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    response = openai.ChatCompletion.create(
        engine="gptauthor",
        messages=[
            {"role": "system", "content": "You are a creative assistant that writes a spec for the cover art of a book, based on the book's plot."},
            {"role": "user", "content": f"Plot: {plot}\n\n--\n\nDescribe the cover we should create, based on the plot. This should be two sentences long, maximum."}
        ]
    )

    save_content(cacheName, response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']

def create_image(imageName, inputPrompt):
    repo_id = "./stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(repo_id)

    pipe = pipe.to("cuda")

    image = pipe(inputPrompt).images[0]

    image.save("./content/" + imageName + ".png")

    return image

def create_cover_image(plot):
    cacheName = "cover"

    file_path = "./content/" + cacheName + ".png"
    if os.path.exists(file_path):
        return Image.open(file_path)

    plot = str(generate_cover_prompt(plot))

    create_image("cover",plot)


def create_epub(title, author, chapters, chapter_images, cover_image_path='./content/cover.png'):
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier('id123456')
    book.set_title(title)
    book.set_language('en')
    book.add_author(author)

    # Add cover image
    with open(cover_image_path, 'rb') as cover_file:
        cover_image = cover_file.read()
    book.set_cover('cover.png', cover_image)

    # Create chapters and add them to the book
    epub_chapters = []
    for i, (chapter_dict, chapter_image) in enumerate(zip(chapters, chapter_images)):
        full_chapter_title = list(chapter_dict.keys())[0]
        chapter_content = list(chapter_dict.values())[0]
        if ' - ' in full_chapter_title:
            chapter_title = full_chapter_title.split(' - ')[1]
        else:
            chapter_title = full_chapter_title

        chapter_file_name = f'chapter_{i+1}.xhtml'
        epub_chapter = epub.EpubHtml(
            title=chapter_title, file_name=chapter_file_name, lang='en')
        
        # Convert chapter image to bytes
        image_stream = BytesIO()
        chapter_image.save(image_stream, format='JPEG')
        image_data = image_stream.getvalue()

        # Embed image in the XHTML content
        image_base64 = base64.b64encode(image_data).replace(b'\n', b'')
        image_tag = f'<img src="data:image/jpeg;base64,{image_base64.decode()}" />'

        # Add paragraph breaks
        formatted_content = ''.join(
            f'<p>{paragraph.strip()}</p>' for paragraph in chapter_content.split('\n') if paragraph.strip())

        epub_chapter.content = f'<h1>{chapter_title}</h1>{formatted_content}'
        book.add_item(epub_chapter)
        epub_chapters.append(epub_chapter)

    # Define Table of Contents
    book.toc = (epub_chapters)

    # Add default NCX and Nav files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define CSS style
    style = '''
    @namespace epub "http://www.idpf.org/2007/ops";
    body {
        font-family: Cambria, Liberation Serif, serif;
    }
    h1 {
        text-align: left;
        text-transform: uppercase;
        font-weight: 200;
    }
    '''

    # Add CSS file
    nav_css = epub.EpubItem(
        uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)

    # Create spine
    book.spine = ['nav'] + epub_chapters

    # Save the EPUB file
    epub.write_epub(f'{title}.epub', book)

def print_step_costs(response, model):
    input = response['usage']['prompt_tokens']
    output = response['usage']['completion_tokens']

    if model == "gpt-4" or model == "gpt-4-0613":
        input_per_token = 0.00003
        output_per_token = 0.00006
    if model == "gpt-3.5-turbo-16k":
        input_per_token = 0.000003
        output_per_token = 0.000004
    if model == "gpt-4-32k-0613" or model == "gpt-4-32k":
        input_per_token = 0.00006
        output_per_token = 0.00012
    if model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-0613":
        input_per_token = 0.0000015
        output_per_token = 0.000002

    input_cost = int(input) * input_per_token
    output_cost = int(output) * output_per_token

    total_cost = input_cost + output_cost
    print('step cost:', total_cost)


def generate_plots(prompt):
    cacheName = "generate_plots"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content.split('\n')

    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are a creative assistant that generates engaging novel plots."},
            {"role": "user", "content": f"Generate 10 novel plots based on this prompt: {prompt}"}
        ]
    )

    print_step_costs(response, "gpt-4-0613")

    save_content(cacheName, response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content'].split('\n')


def select_most_engaging(plots):
    cacheName = "select_most_engaging"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are an expert in writing plots for books."},
            {"role": "user", "content": f"Here are a number of possible plots for a new novel: {plots}\n\n--\n\nNow, write the final plot that we will go with. It can be one of these, a mix of the best elements of multiple, or something completely new and better. The most important thing is the plot should be fun, unique, and engaging."}
        ]
    )

    print_step_costs(response, "gpt-4-0613")

    save_content(cacheName, response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']


def improve_plot(plot):
    cacheName = "improve_plot"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are an expert in improving and refining story plots."},
            {"role": "user", "content": f"Improve this plot: {plot}"}
        ]
    )

    print_step_costs(response, "gpt-4-0613")

    save_content(cacheName, response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']


def get_title(plot):
    cacheName = "get_title"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are an expert writer."},
            {"role": "user", "content": f"Here is the plot: {plot}\n\nWhat is the title of this book? Just respond with the title, do nothing else. Write the title in {chosenLanguage}."}
        ]
    )

    print_step_costs(response, "gpt-3.5-turbo-16k")

    save_content(cacheName, response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']


def write_first_chapter(plot, first_chapter_title, writing_style):
    cacheName = "write_first_chapter"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are a world-class writer."},
            {"role": "user", "content": f"Here is the high-level plot to follow: {plot}\n\nWrite the first chapter of this novel: `{first_chapter_title}`.\n\nMake it incredibly unique, engaging, and well-written.\n\nHere is a description of the writing style you should use: `{writing_style}`\n\nInclude only the chapter text. There is no need to rewrite the chapter name. Write it in {chosenLanguage}."}
        ]
    )

    print_step_costs(response, "gpt-4-0613")

    improved_response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are a world-class writer. Your job is to take your student's rough initial draft of the first chapter of their novel, and rewrite it to be significantly better, with much more detail."},
            {"role": "user",
                "content": f"Here is the high-level plot you asked your student to follow: {plot}\n\nHere is the first chapter they wrote: {response['choices'][0]['message']['content']}\n\nNow, rewrite the first chapter of this novel, in a way that is far superior to your student's chapter. It should still follow the exact same plot, but it should be far more detailed, much longer, and more engaging. Here is a description of the writing style you should use: `{writing_style}`. Write in {chosenLanguage}."}
        ]
    )

    print_step_costs(response, "gpt-4-32k-0613")

    save_content(cacheName, response['choices'][0]['message']['content'])

    return improved_response['choices'][0]['message']['content']

def get_chapter_image(plot, chapter_title, chapterNumber):
    cacheName = "write_chapter_image_" + str(chapterNumber)

    file_path = "./content/" + cacheName + ".png"
    if os.path.exists(file_path):
        return Image.open(file_path)

    # Get image title
    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are a creative assistant that writes the specification for images given chapter summaries and titles."},
            {"role": "user", "content": f"Plot: {plot}\n\n--\n\nDescribe the chapter image we should create, based on the plot and title: {chapter_title}. This should be two sentences long, maximum."}
        ]
    )

    save_content(cacheName, response['choices'][0]['message']['content'])

    prompt = response['choices'][0]['message']['content']

    create_image(cacheName,prompt)

def write_chapter(previous_chapters, plot, chapter_title, chapterNumber):
    cacheName = "write_chapter_" + str(chapterNumber)

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    try:
        i = random.randint(1, 2242)
        # write_to_file(f'write_chapter_{i}', f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name.")
        response = openai.ChatCompletion.create(
            engine=engineName,
            messages=[
                {"role": "system", "content": "You are a world-class writer."},
                {"role": "user", "content": f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name. Write it in {chosenLanguage}."}
            ]
        )

        print_step_costs(response, "gpt-4-0613")

        save_content(cacheName, response['choices'][0]['message']['content'])

        return response['choices'][0]['message']['content']
    except:
        response = openai.ChatCompletion.create(
            engine=engineName,
            messages=[
                {"role": "system", "content": "You are a world-class writer."},
                {"role": "user", "content": f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name. Write it in {chosenLanguage}."}
            ]
        )

        print_step_costs(response, "gpt-4-32k-0613")

        save_content(cacheName, response['choices'][0]['message']['content'])

        return response['choices'][0]['message']['content']


def generate_storyline(prompt, num_chapters):
    cacheName = "generate_storyline"

    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    print("Generating storyline with chapters and high-level details...")
    json_format = """[{"Chapter CHAPTER_NUMBER_HERE - CHAPTER_TITLE_GOES_HERE": "CHAPTER_OVERVIEW_AND_DETAILS_GOES_HERE"}, ...]"""
    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are a world-class writer. Your job is to write a detailed storyline, complete with chapters, for a novel. Don't be flowery -- you want to get the message across in as few words as possible. But those words should contain lots of information."},
            {"role": "user", "content": f'Write an engaging storyline with {num_chapters} chapters and high-level details based on this plot: {prompt}.\n\nDo it in this list of dictionaries format {json_format}'}
        ]
    )

    print_step_costs(response, "gpt-4-0613")

    improved_response = openai.ChatCompletion.create(
        engine=engineName,
        messages=[
            {"role": "system", "content": "You are a world-class writer. Your job is to take your student's rough initial draft of the storyline of a novel, and rewrite it to be significantly better."},
            {"role": "user", "content": f"Here is the draft storyline they wrote: {response['choices'][0]['message']['content']}\n\nNow, rewrite the storyline, in a way that is far superior to your student's version. It should have the same number of chapters, but it should be much improved in as many ways as possible. The output MUST ONLY include a list of dictionaries format like so: {json_format} do not include any other text besides that."}
        ]
    )

    print_step_costs(improved_response, "gpt-4-0613")

    save_content(cacheName, response['choices'][0]['message']['content'])

    return improved_response['choices'][0]['message']['content']


def write_to_file(prompt, content):

    # Create a directory for the prompts if it doesn't exist
    if not os.path.exists('prompts'):
        os.mkdir('prompts')

    # Replace invalid characters for filenames
    valid_filename = ''.join(
        c for c in prompt if c.isalnum() or c in (' ', '.', '_')).rstrip()
    file_path = f'prompts/{valid_filename}.txt'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'Output for prompt "{prompt}" has been written to {file_path}\n')


def write_novel(prompt, num_chapters, writing_style):
    plots = generate_plots(prompt)

    best_plot = select_most_engaging(plots)

    improved_plot = improve_plot(best_plot)

    title = get_title(improved_plot)

    storyline = generate_storyline(improved_plot, num_chapters)
    chapter_titles = ast.literal_eval(storyline)

    novel = f"Storyline:\n{storyline}\n\n"

    first_chapter = write_first_chapter(
        storyline, chapter_titles[0], writing_style.strip())
    novel += f"Chapter 1:\n{first_chapter}\n"
    chapters = [first_chapter]
    chapter_images = [get_chapter_image(first_chapter,chapter_titles[0],0)]

    for i in range(num_chapters - 1):
        # + 2 because the first chapter was already added
        print(f"Writing chapter {i+2}...")
        chapter = write_chapter(novel, storyline, chapter_titles[i+1], i+1)
        novel += f"Chapter {i+2}:\n{chapter}\n"
        chapters.append(chapter)
        chapter_image = get_chapter_image(storyline, chapter_titles[i+1],i+1)
        chapter_images.append(chapter_image)

    return novel, title, chapters, chapter_titles, chapter_images

def mainFunc():
    # Example usage:
    prompt = "A lion gets lost in Seoul and finds the city's most famous places."
    num_chapters = 5
    writing_style = "Clear and easily understandable, similar to a young adult novel. Highly descriptive and sometimes long-winded."
    novel, title, chapters, chapter_titles, chapter_images = write_novel(
        prompt, num_chapters, writing_style)

    # Replace chapter descriptions with body text in chapter_titles
    for i, chapter in enumerate(chapters):
        chapter_number_and_title = list(chapter_titles[i].keys())[0]
        chapter_titles[i] = {chapter_number_and_title: chapter}

    # Create the cover
    create_cover_image(str(chapter_titles))

    # Create the EPUB file
    create_epub(title, 'AI', chapter_titles, chapter_images, './content/cover.png')

if __name__ == "__main__":
    mainFunc()

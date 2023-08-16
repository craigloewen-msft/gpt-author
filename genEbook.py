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
totalCostTracker = 0

# get it at https://beta.dreamstudio.ai/
stability_api_key = "ENTER STABILITY KEY HERE"

# Helper functions

def print_step_costs(response, model):
    global totalCostTracker

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

    step_total_cost = input_cost + output_cost
    totalCostTracker += step_total_cost
    print('step cost:', step_total_cost, ' total cost:', totalCostTracker)

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

def callAICompletion(cacheName,messages):
    cached_content = check_and_load_cache(cacheName)
    if cached_content:
        return cached_content

    response = openai.ChatCompletion.create(
        engine=engineName,
        messages=messages
    )

    print_step_costs(response, "gpt-4-0613")

    responseText = response['choices'][0]['message']['content']

    save_content(cacheName, responseText)

    return responseText

def create_image(imageName, inputPrompt):
    cacheName = imageName

    file_path = "./content/" + cacheName + ".png"
    if os.path.exists(file_path):
        return Image.open(file_path)
    
    repo_id = "./stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(repo_id)

    pipe = pipe.to("cuda")

    image = pipe(inputPrompt).images[0]

    image.save("./content/" + imageName + ".png")

    return image

def get_chapter_image(plot, chapter_title, chapterNumber):
    cacheName = "write_chapter_image_" + str(chapterNumber)

    # Get image title
    response = callAICompletion(cacheName,[
            {"role": "system", "content": "You are a creative assistant that writes the specification for images given chapter summaries and titles."},
            {"role": "user", "content": f"Plot: {plot}\n\n--\n\nDescribe the chapter image we should create, based on the plot and title: {chapter_title}. This should be two sentences long, maximum and always in a digital art style."}
        ]
    )

    return create_image(cacheName, response)

# Functions used in the main process

def create_cover_image(plot):
    cacheName = "cover"

    plot = str(generate_cover_prompt(plot))

    return create_image("cover", plot)

def generate_cover_prompt(plot):
    
    response = callAICompletion("generate_cover_prompt",[
            {"role": "system", "content": "You are a creative assistant that writes a spec for the cover art of a book, based on the book's plot."},
            {"role": "user", "content": f"Plot: {plot}\n\n--\n\nDescribe the cover we should create, based on the plot. This should be two sentences long, maximum."}
        ]
    )
    return response

def create_epub(title, author, chapters, chapter_images, cover_image_path='./content/cover.png'):
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier('id123456')
    book.set_title(title)
    book.set_language('ko')
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

        full_chapter_content = f'{image_tag}<h1>{chapter_title}</h1>{formatted_content}'

        epub_chapter.content = full_chapter_content
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

def generate_plots(learning_goal):
    response = callAICompletion("generate_plots",[
            {"role": "system", "content": "You are a creative assistant that generates engaging novel plots."},
            {"role": "user", "content": f"Generate 10 novel plots that will tell a fun, unique and engaging story that will also teach the reader about this topic: {learning_goal}"}
        ]
    )

    return response.split('\n')

def select_most_engaging(plots):
    response = callAICompletion("select_most_engaging",[
            {"role": "system", "content": "You are an expert in writing plots for books."},
            {"role": "user", "content": f"Here are a number of possible plots for a new novel: {plots}\n\n--\n\nNow, write the final plot that we will go with. It can be one of these, a mix of the best elements of multiple, or something completely new and better. The most important thing is the plot should be fun, unique, and engaging."}
        ]
    )

    return response

def improve_plot(plot):
    response = callAICompletion("improve_plot",[
            {"role": "system", "content": "You are an expert in improving and refining story plots."},
            {"role": "user", "content": f"Improve this plot: {plot}"}
        ]
    )

    return response

def get_title(plot):
    response = callAICompletion("get_title",[
            {"role": "system", "content": "You are an expert writer."},
            {"role": "user", "content": f"Here is the plot: {plot}\n\nWhat is the title of this book? Just respond with the title, do nothing else. The title should only be outputted in {chosenLanguage} and should not include any pronounciation guides."}
        ]
    )

    return response

def write_vocab(inputText, chapterNumber):
    response = callAICompletion("vocab_chapter_" + str(chapterNumber), [
            {"role": "system", "content": f"You are a world-class {chosenLanguage} tutor."},
            {"role": "user", "content": f"Please write out the necessary vocabulary with translations (and no pronounciation guides) for the given chapter of a book. It should be a maximum of 10 words long and only include the most important words and should be a list in this format (Vocab word): (English Translation).\n\n Book Chapter:{inputText}"}
        ]
    )
    return response

def write_first_chapter(plot, first_chapter_title, writing_style):
    response = callAICompletion("write_first_chapter",[
            {"role": "system", "content": "You are a world-class writer."},
            {"role": "user", "content": f"Here is the high-level plot to follow: {plot}\n\nWrite the first chapter of this novel: `{first_chapter_title}`.\n\nMake it incredibly unique, engaging, and well-written.\n\nHere is a description of the writing style you should use: `{writing_style}`\n\nInclude only the chapter text. There is no need to rewrite the chapter name. Write it in {chosenLanguage}."}
        ]
    )

    improved_response = callAICompletion("improved_response",[
            {"role": "system", "content": "You are a world-class writer. Your job is to take your student's rough initial draft of the first chapter of their novel, and rewrite it to be significantly better, with much more detail."},
            {"role": "user",
                "content": f"Here is the high-level plot you asked your student to follow: {plot}\n\nHere is the first chapter they wrote: {response}\n\nNow, rewrite the first chapter of this novel, in a way that is far superior to your student's chapter. It should still follow the exact same plot, but it should be far more detailed, much longer, and more engaging. Here is a description of the writing style you should use: `{writing_style}`. Write in {chosenLanguage}."}
        ]
    )

    chapterContent = improved_response

    getChapterVocab = write_vocab(chapterContent, 0)

    return chapterContent, getChapterVocab

def write_chapter(previous_chapters, plot, chapter_title, chapterNumber):
    chapterContent = callAICompletion("write_chapter_" + str(chapterNumber),[
            {"role": "system", "content": "You are a world-class writer."},
            {"role": "user", "content": f"Plot: {plot}, Previous Chapters: {previous_chapters}\n\n--\n\nWrite the next chapter of this novel, following the plot and taking in the previous chapters as context. Here is the plan for this chapter: {chapter_title}\n\nWrite it beautifully. Include only the chapter text. There is no need to rewrite the chapter name. Write it in {chosenLanguage}."}
        ]
    )

    getChapterVocab = write_vocab(chapterContent, chapterNumber)

    return chapterContent, getChapterVocab

def generate_storyline(prompt, num_chapters, learning_goal):
    json_format = """[{"Chapter CHAPTER_NUMBER_HERE - CHAPTER_TITLE_GOES_HERE": "CHAPTER_OVERVIEW_AND_DETAILS_GOES_HERE"}, ...]"""
    response = callAICompletion("generate_storyline",[
            {"role": "system", "content": "You are a world-class writer. Your job is to write a detailed storyline, complete with chapters, for a novel. Don't be flowery -- you want to get the message across in as few words as possible. But those words should contain lots of information."},
            {"role": "user", "content": f'Write an engaging storyline with {num_chapters} chapters and high-level details based on this plot: {prompt}. It should also teach the user about {learning_goal}.\n\nDo it in this list of dictionaries format {json_format}'}
        ]
    )

    improved_response = callAICompletion("generate_storyline_improved",[
            {"role": "system", "content": "You are a world-class writer. Your job is to take your student's rough initial draft of the storyline of a novel, and rewrite it to be significantly better."},
            {"role": "user", "content": f"Here is the draft storyline they wrote: {response}\n\nNow, rewrite the storyline, in a way that is far superior to your student's version. It should have the same number of chapters, but it should be much improved in as many ways as possible. The output MUST ONLY include a list of dictionaries format like so: {json_format} do not include any other text besides that."}
        ]
    )

    return improved_response

def write_novel(learning_goal, num_chapters, writing_style):

    print("Making plots")
    plots = generate_plots(learning_goal)

    print("Choosing best plot")
    best_plot = select_most_engaging(plots)

    print("Improving best plot")
    improved_plot = improve_plot(best_plot)

    print("Generating Title")
    title = get_title(improved_plot)

    print("Making storyline")
    storyline = generate_storyline(improved_plot, num_chapters, learning_goal)
    chapter_titles = ast.literal_eval(storyline)

    novel = f"Storyline:\n{storyline}\n\n"

    print("Writing chapter 1...")
    first_chapter, first_vocab = write_first_chapter(
        storyline, chapter_titles[0], writing_style.strip())
    novel += f"Chapter 1:\n{first_chapter}\n"
    chapters = [first_chapter]
    vocab_chapter_list = [first_vocab]
    chapter_images = [get_chapter_image(first_chapter, chapter_titles[0], 0)]

    for i in range(num_chapters - 1):
        # + 2 because the first chapter was already added
        print(f"Writing chapter {i+2}...")
        chapter, vocab = write_chapter(novel, storyline, chapter_titles[i+1], i+1)
        novel += f"Chapter {i+2}:\n{chapter}\n"
        chapters.append(chapter)
        vocab_chapter_list.append(vocab)
        chapter_image = get_chapter_image(storyline, chapter_titles[i+1], i+1)
        chapter_images.append(chapter_image)

    combinedChapterContent = []

    for i in range(len(chapters)):
        combinedChapterContent.append(vocab_chapter_list[i] + "\n\n" + chapters[i])
        
    return novel, title, combinedChapterContent, chapter_titles, chapter_images

def mainFunc():
    # Example usage:
    learning_goal = "The city Seoul"
    num_chapters = 10
    writing_style = "Clear and easily understandable, similar to a young adult novel. Highly descriptive and sometimes long-winded."
    novel, title, chapters, chapter_titles, chapter_images = write_novel(
        learning_goal, num_chapters, writing_style)

    # Replace chapter descriptions with body text in chapter_titles
    for i, chapter in enumerate(chapters):
        chapter_number_and_title = list(chapter_titles[i].keys())[0]
        chapter_titles[i] = {chapter_number_and_title: chapter}

    # Create the cover
    create_cover_image(str(chapter_titles))

    # Create the EPUB file
    create_epub(title, 'AI', chapter_titles,
                chapter_images, './content/cover.png')


if __name__ == "__main__":
    mainFunc()

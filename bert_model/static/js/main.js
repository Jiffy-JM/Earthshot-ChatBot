

// INTERACTIONS SECTION

var chatButton = document.getElementById('chatbot');
var injectionSection = document.getElementById('injection_section');
var chatIcon = document.getElementById('chatIcon');
var closeButton = document.getElementById('closeButton');
var messageSection = document.getElementById('message_section');
var chatTitle = document.getElementById('chatTitle');
var sendMessage = document.getElementById('sendMessage');
var suggestions = document.getElementById('suggestions');
var opened = false;


chatButton.addEventListener('click', async(e) => {

    if(opened === false){
        opened = true;
        chatButton.style.width = '300px';
        chatButton.style.height = '350px';
        chatButton.style.borderRadius = '14px';
        chatButton.style.top = '50%';
        chatIcon.style.opacity = '0';
        chatButton.style.textAlign = 'none';
        injectionSection.style.display = 'flex'

        chatIcon.style.display = 'none';

        closeButton.style.display = 'inline';

        messageSection.style.display = 'flex';

        chatButton.style.cursor = 'default'

        chatTitle.style.display = 'inline';

        
    }
    
});

closeButton.addEventListener('click', async(e) => {
    injectionSection.style.display = 'none';
    chatButton.style.width = '52px';
    chatButton.style.height = '52px';
    chatButton.style.borderRadius = '100px';
    chatButton.style.top = '80%';
    chatIcon.style.opacity = '1';
    chatButton.style.textAlign = 'center';
    
    chatIcon.style.display = 'inline';

    closeButton.style.display = 'none';

    messageSection.style.display = 'none';

    chatButton.style.cursor = 'pointer'
    
    chatTitle.style.display = 'none';

    setTimeout(async function(){
        opened = false;
    },500);
});

var message_id = 1;
sendMessage.addEventListener('click', async(e) => {
    var curr_message = document.getElementById('messageInput').value;
    var curr_response;
    console.log(curr_message)

    // if not empty, sned to API
    if(curr_message !== "" && curr_message !== " " && curr_message !== null){
        //clear suggs
        suggestions.style.display = 'none';
        
        //clear message
        document.getElementById('messageInput').value = "";
        //loading animation
        sendMessage.innerHTML = `<svg class="spinner" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.4.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path d="M304 48a48 48 0 1 0 -96 0 48 48 0 1 0 96 0zm0 416a48 48 0 1 0 -96 0 48 48 0 1 0 96 0zM48 304a48 48 0 1 0 0-96 48 48 0 1 0 0 96zm464-48a48 48 0 1 0 -96 0 48 48 0 1 0 96 0zM142.9 437A48 48 0 1 0 75 369.1 48 48 0 1 0 142.9 437zm0-294.2A48 48 0 1 0 75 75a48 48 0 1 0 67.9 67.9zM369.1 437A48 48 0 1 0 437 369.1 48 48 0 1 0 369.1 437z"/></svg>`

        //disable button
        sendMessage.disabled = true;

        // show user message
        injectionSection.innerHTML += `<div class="userMessage">${curr_message}</div>`

        //send to API
        curr_response = await sendToChat(curr_message);

        //show response message
        injectionSection.innerHTML += `<div class="responseMessage">${curr_response}</div>`

        //end animation
        sendMessage.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.4.0 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2023 Fonticons, Inc. --><path d="M498.1 5.6c10.1 7 15.4 19.1 13.5 31.2l-64 416c-1.5 9.7-7.4 18.2-16 23s-18.9 5.4-28 1.6L284 427.7l-68.5 74.1c-8.9 9.7-22.9 12.9-35.2 8.1S160 493.2 160 480V396.4c0-4 1.5-7.8 4.2-10.7L331.8 202.8c5.8-6.3 5.6-16-.4-22s-15.7-6.4-22-.7L106 360.8 17.7 316.6C7.1 311.3 .3 300.7 0 288.9s5.9-22.8 16.1-28.7l448-256c10.7-6.1 23.9-5.5 34 1.4z"/></svg>`
        
        //scroll
        injectionSection.scrollBy({top:message_id * 1200, behavior: 'smooth'});
        message_id += 1;
        //enable button
        sendMessage.disabled = false;
    }
    


});

//click a suggestion
suggestions.addEventListener('click', async(e) => {
    var clickedSugg = e.target.closest('p').innerText;
    suggestions.style.display = 'none';
    document.getElementById('messageInput').value = clickedSugg;
    sendMessage.click()
});


// API SECTION // injection 


 // SEND TEXT TO GPT-3.5-TURBO --> no longer in use --> using sendToChat now
 async function sendToGPT(query){

    let txt;
    //call openai api
    await fetch("https://api.openai.com/v1/chat/completions", { //we will replace this link with our API
        method: "POST",
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer API_KEY_HERE'
        },
        body: JSON.stringify({
            "model": "gpt-3.5-turbo",           // LIST OF MODELS (format changes slightly for each)
            "messages": [{"role":"user","content":query}],       //      gpt-3.5-turbo = chatGPT (as of 3/26/2023)
            "temperature": 0.35,
            "max_tokens": 500,                                  // this is OpenAI API call structure. most of this will change/ go away
            "top_p": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        })
        }
    ).then(response => response.json())
    .then(data => {
        //RAW
        txt = data['choices'][0]['message']['content']
    })
    .catch(error => {
        txt = error
    })

    console.log(txt)

    return txt;
}

// request responds from python script
async function sendToChat(query){
    let txt;

    // send an http request to python script
    await fetch('/process-request', {
        method:'POST',
        headers: {
            'Content-Type':'application/json'
        },
        body: JSON.stringify(query)
        
    }).then((response) => {
        console.log(response)
        return response.json()
    })
    .then((myJson) => {
        console.log(myJson.message); 
        txt = myJson.message
    })
    .catch((error) => {
        console.log(error)
        txt = error
    })

    return txt
}
/*.style.display = 'block';*/
let firstwidth;
window.onresize = reportWindowSize;

window.onload = function() {
    firstwidth =  window.innerWidth;
    let aID = document.title.replaceAll(' ', '_')
    document.getElementById(aID).classList.add("chosen");
};

function reportWindowSize(){
    let width = window.innerWidth;
    let x = document.querySelectorAll(".navL");
    /* if user resized to desktop view from mobile view*/
    if((width > 768) && (firstwidth < 768)){
        for (let i = 0; i < x.length; i++){
            x[i].style.display='inline-block';
        }
    }
    /* if user resized from desktop view to mobile view*/
    if((width < 768) && (firstwidth > 768)){
        for (let i = 0; i < x.length; i++){
            x[i].style.display='none';
        }
    }
    firstwidth =  window.innerWidth;
}

function openMenu() {
    let x = document.querySelectorAll(".navL");
    if(x[0].style.display==='inline-block'){
        for (let i = 0; i < x.length; i++){
            x[i].style.display='none';
        }
    }
    else{
        for (let i = 0; i < x.length; i++){
            x[i].style.display='inline-block';
        }
    }
}
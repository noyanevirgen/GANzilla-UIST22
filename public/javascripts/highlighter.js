var x = 0
var seed = []
var flask_url = "http://localhost:5000/"
var ui_url = "http://localhost:3000/"
$.getJSON( "state.txt", function( data ) {
    seed = data['seed']
    
});

function get_image(input){
    var jqXHR = $.ajax({
        type: "POST",
        url: flask_url + "get_image",
        data: { mydata: JSON.stringify(input) },
        async: false,
    });
    return jqXHR.responseText;
  } 

var paint = Painterro({id: "Highlight",
    saveHandler: function (image, done) {
        var formData = new FormData();
        formData.append('image', image.asBlob());
        // you can also pass suggested filename 
        // formData.append('image', image.asBlob(), image.suggestedFileName());
        var xhr = new XMLHttpRequest();
        xhr.open('POST', flask_url + 'save-as-binary/', true);
        xhr.onload = xhr.onerror = function () {
        // after saving is done, call done callback
        done(false); //done(true) will hide painterro, done(false) will leave opened
        };
        xhr.send(formData);
        console.log(seed);
        if(x < seed.length-1){
            x += 1
            k = JSON.parse(get_image(x))
            paint.show(k)
        }
        else {
            //window.location.href = "http://localhost:3000/select";
            console.log('sad'); //json output 
        }
    },

    how_to_paste_actions: ['replace_all'],
    defaultTool: 'brush',
    backgroundFillColor: '#222',
    defaultSize: '512x512'
}).show(JSON.parse(get_image(0)))



$('.skiptoselectbutton').click(function (e) {
    window.location.href = ui_url + "select";
});

function highlighter_opens(input){
    var jqXHR = $.ajax({
        type: "POST",
        url: flask_url + "highlighter_opens",
        data: { mydata: JSON.stringify(input) },
        success: first_executed,
    });
  } 

function first_executed(response) {
}
function codeAddress() {
    highlighter_opens();
  }
  window.onload = codeAddress;




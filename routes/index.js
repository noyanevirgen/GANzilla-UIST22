var express = require('express');
var router = express.Router();

/* GET home page. */

router.get('/highlight', function(req, res, next) {
  res.render('first', { title: 'Highlight' });
});

router.get('/select', function(req, res, next) {
  res.render('select', { title: 'Select' });
});

router.get('/ganzilla', function(req, res, next) {
  res.render('index', { title: 'GANzilla' });
});

router.get('/index.htm', function (req, res) {
  res.sendFile( __dirname + "/" + "index.htm" );
})

router.get('/process_get', function (req, res) {
  // Prepare output in JSON format
  response = {
     first_name:req.query.first_name,
     last_name:req.query.last_name
  };
  console.log(response);
  res.json(JSON.stringify(response));
})

module.exports = router;

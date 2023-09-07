//jshint esversion:6
require("dotenv").config();
const express = require("express");
const bodyParser = require("body-parser");
const ejs = require("ejs");
const mongoose = require('mongoose');
const encrypt = require("mongoose-encryption");

const app = express();

app.set('view engine', 'ejs');

app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(express.static("public"));

mongoose.connect("mongodb://localhost:27017/userDB", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  family: 4, // Force IPv4
});

const userSchema = new mongoose.Schema({
    email: String,
    password: String
});


userSchema.plugin(encrypt, { secret: process.env.SECRET, encryptedFields: ["password"] }) ;
 
const User = new mongoose.model("User", userSchema);

app.get("/", function(req, res) {
    res.render("home");
})

app.get("/login", function(req, res) {
    res.render("login");
})

app.get("/register", function(req, res) {
    res.render("register");
})

app.post("/register", async (req, res) => {
    const { username, password } = req.body;

    try {
        const newUser = new User({
            email: username,
            password: password
        });

        await newUser.save();
        console.log("User registered successfully");
        
        // Render the "secrets" view upon successful registration
        res.render("secrets");
    } catch (error) {
        console.error("Error registering user:", error);
        // You can send an error response to the client if the save operation fails
        res.status(500).send("Error registering user");
    }
});


app.post("/login", async (req, res) => {
    const username = req.body.username;
    const password = req.body.password;

    try {
        const foundUser = await User.findOne({ email: username }).exec();

        if (foundUser) {
            if (foundUser.password === password) {
                // Render the "secrets" view/page upon successful login
                return res.render("secrets");
            }
        }

        // Handle the case when the user is not found or the password is incorrect
        // You can send an appropriate response to the client.
        res.status(401).send("Authentication failed");
    } catch (error) {
        console.error("Error during login:", error);
        // Handle the error here, e.g., send an error response to the client.
        res.status(500).send("Error during login");
    }
});



app.listen(3000, function() {
  console.log("Server started on port 3000");
});
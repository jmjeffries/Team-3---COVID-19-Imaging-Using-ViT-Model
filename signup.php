
// Database configuration
$servername = "localhost";
$username = "root";
$password = ""; 
$dbname = "UserDatabase";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Retrieve form data
$user = $_POST['username'];
$pass = password_hash($_POST['password'], PASSWORD_BCRYPT); // Secure password hashing

// Insert data into the database
$sql = "INSERT INTO Users (username, password) VALUES ('$user', '$pass')";

if ($conn->query($sql) === TRUE) {
    echo "New record created successfully";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

// Close connection
$conn->close();

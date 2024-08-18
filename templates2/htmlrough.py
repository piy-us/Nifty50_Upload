<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AAPL Stock Prediction</title>
  <!-- Bootstrap CSS -->
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>

  <!-- Navigation Bar -->
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <a class="navbar-brand" href="#">Stock Prediction</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item active">
          <a class="nav-link" href="#">Home <span class="sr-only">(current)</span></a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">About</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Services</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="#">Contact</a>
        </li>
      </ul>
    </div>
  </nav>

  <!-- corousal -->
  <div id="carouselExample" class="carousel slide">
    <div class="carousel-inner">
      <div class="carousel-item active">
        <img src="/templates/img/1.jpg" class="d-block w-100" alt="...">
      </div>
      <div class="carousel-item">
        <img src="..." class="d-block w-100" alt="...">
      </div>
      <div class="carousel-item">
        <img src="..." class="d-block w-100" alt="...">
      </div>
    </div>
    <button class="carousel-control-prev" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
      <span class="carousel-control-prev-icon" aria-hidden="true"></span>
      <span class="visually-hidden">Previous</span>
    </button>
    <button class="carousel-control-next" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
      <span class="carousel-control-next-icon" aria-hidden="true"></span>
      <span class="visually-hidden">Next</span>
    </button>
  </div>


  <br><br>

  <div class="container mt-4">
    <h1>AAPL Stock Prediction</h1>
    <div class="card-deck">
      <div class="card">
        <h2 class="card-title">Plot</h2>
        <a class="card-link" href="/plot">View Plot</a>
      </div>
      <div class="card">
        <h2 class="card-title">Performance</h2>
        <a class="card-link" href="/performance">View Performance</a>
      </div>
      <div class="card">
        <h2 class="card-title">Forecast</h2>
        <div class="card-body">
          <form action="/forecast" method="post">
            <div class="form-group">
              <label for="num_days">Number of Days:</label>
              <input type="text" class="form-control" id="num_days" name="num_days">
            </div>
            <button type="submit" class="btn btn-primary">Generate Forecast</button>
          </form>
        </div>
      </div>
    </div>
  </div>


  <!-- Bootstrap JS and jQuery -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>


<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AAPL Stock Prediction</title>
  <!-- Bootstrap CSS -->
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
  <div class="container mt-4">
    <h1>AAPL Stock Prediction</h1>
    <div class="card-deck">
      <div class="card">
        <h2 class="card-title">Plot</h2>
        <a class="card-link" href="/plot">View Plot</a>
      </div>
      <div class="card">
        <h2 class="card-title">Performance</h2>
        <a class="card-link" href="/performance">View Performance</a>
      </div>
      <div class="card">
        <h2 class="card-title">Forecast</h2>
        <div class="card-body">
          <form action="/forecast" method="post">
            <div class="form-group">
              <label for="num_days">Number of Days:</label>
              <input type="text" class="form-control" id="num_days" name="num_days">
            </div>
            <button type="submit" class="btn btn-primary">Generate Forecast</button>
          </form>
        </div>
      </div>
    </div>
  </div>

  <!-- Bootstrap JS and jQuery -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>

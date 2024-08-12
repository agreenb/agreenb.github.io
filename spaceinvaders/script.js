// Potential add-ons:
// - user shoots at invaders
// - keep and display score
// - different invader rows give different points when hit
// - power-ups dropped from invaders,
//   e.g. extra points, temp increased user bullet speed, temp user invincibility

$(function () {
  var canvas = $("#canvas")[0];
  var ctx = canvas.getContext("2d");

  var score_div = $("#score");

  var canvas_width = 440;
  var canvas_height = 440;

  $("#canvas").attr("width", canvas_width);
  $("#canvas").attr("height", canvas_height);

  // How to set up the grid.
  const cell_size_factor = 20;
  const cell_size = canvas_width / cell_size_factor;
  const num_invader_rows = 3;
  const num_invader_cols = 11;
  const cell_padding = 10;
  const max_col = canvas_width / (cell_size + cell_padding) - 1;
  const max_row = max_col;

  // Tracking game updates.
  var refresh_interval;
  const refresh_rate_ms = 100;
  var end_of_game = true;
  var curr_invader_time_tick = 0;
  var max_invader_time_tick = 4;
  var score = 0;

  // Tracking user, invaders, and bullets from invaders.
  var user_direction = "";
  var prev_invader_dir = "right";
  var invader_dir = "right";
  var invaders = [];
  var user_col = 0;
  var bullets = [];
  var bullet_width = 5;
  var bullet_height = 5;

  function init_screen() {
    // Reset variables.
    user_direction = "";
    user_col = 0;
    invaders = [];
    prev_invader_dir = "right";
    invader_dir = "right";
    bullets = [];
    score = 0;

    // Add all invaders.
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas_width, canvas_height);
    for (var col = 0; col < num_invader_cols; col++) {
      var invader_col = [];
      for (var row = 0; row < num_invader_rows; row++) {
        var cell = { row: row, col: col, color: "green" };
        invader_col.push(cell);
        paint_cell(cell, cell_size, cell_size);
      }
      invaders.push(invader_col);
    }

    // Add user.
    paint_cell(
      { row: max_row, col: user_col, color: "white" },
      cell_size,
      cell_size
    );
  }

  function paint_cell(cell, width, height, is_bullet) {
    ctx.fillStyle = cell.color;
    var x = (cell_size + cell_padding) * cell.col;
    var y = (cell_size + cell_padding) * cell.row;
    // if (is_bullet) {
    //   x += cell_size / 2
    //   y += cell_size * 1.5
    // }
    ctx.fillRect(x, y, width, height);
  }

  $("body").keydown(function (evt) {
    switch (evt.keyCode) {
      case 37: // left
        user_direction = "left";
        break;
      case 39: // right
        user_direction = "right";
        break;
      case 27: // esc
        clearInterval(refresh_interval);
        end_of_game = true;
        score_div.text("Score: 0")
        start();
        break;
      case 13: // enter
        if (end_of_game) {
          end_of_game = false;
          start();
          refresh_interval = setInterval(repaint, refresh_rate_ms);
        }
        break;
      case 32: // space
        if (!end_of_game) {
          user_shoot_at_invader();
          break;
        }
    }
  });

  function repaint() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas_width, canvas_height);

    // Check if the user has pressed a move key.
    switch (user_direction) {
      case "left":
        if (user_col > 0) {
          user_col -= 1;
        }
        break;
      case "right":
        if (user_col < max_col) {
          user_col += 1;
        }
        break;
    }

    // Reset the user movement so that it's only done once.
    user_direction = "";

    // Render the user position, whether it has moved or not.
    paint_cell(
      { row: max_row, col: user_col, color: "white" },
      cell_size,
      cell_size
    );

    // Render the invaders.
    var next_invader_dir = invader_dir;
    for (var col = 0; col < invaders.length; col++) {
      var curr_row = invaders[col];
      for (var i = 0; i < curr_row.length; i++) {
        var invader = curr_row[i];

        // Only update the invaders on their timeframe, which allows
        // the user to move faster than the invaders do.
        if (curr_invader_time_tick == max_invader_time_tick) {
          switch (invader_dir) {
            case "right":
              invader.col += 1;
              if (invader.col >= max_col) {
                next_invader_dir = "down";
              }
              break;
            case "left":
              invader.col -= 1;
              if (invader.col <= 0) {
                next_invader_dir = "down";
              }
              break;
            case "down":
              invader.row += 1;
              if (invader.row >= max_row) {
                game_over();
                break;
              }
              if (prev_invader_dir == "right") {
                next_invader_dir = "left";
              } else {
                next_invader_dir = "right";
              }
              break;
          }
        }

        // Render the invaders, whether they are moving or not.
        if (!end_of_game) {
          paint_cell(invader, cell_size, cell_size);
        }
      }
    }

    if (curr_invader_time_tick == max_invader_time_tick) {
      prev_invader_dir = invader_dir;
      invader_dir = next_invader_dir;
    }

    // Update any bullets from the invader shooting at the user.
    // These also respect the slower refresh rate of the invaders.
    invader_potentially_shoot_bullet();
    move_bullets();

    curr_invader_time_tick += 1;
    curr_invader_time_tick %= max_invader_time_tick + 1;

    score_div.text("Score: " + score)
  }

  function move_bullets() {
    for (var i = bullets.length - 1; i >= 0; i--) {
      var bullet = bullets[i];
      const index = bullets.indexOf(bullet);
      if (bullet.from_user) {
        bullet.row -= 1;
      } else if (curr_invader_time_tick == max_invader_time_tick) {
        bullet.row += 1;
      }

      // If the bullet has reached the bottom, get rid of it.
      if (
        bullet.row * (cell_size + cell_padding) > canvas_height ||
        bullet.row < 0
      ) {
        bullets.splice(index, 1);
      } else if (bullet.from_user) {
        paint_cell(bullet, bullet_width, bullet_height, true);
        if (check_user_bullet_collision_with_invader(bullet)) {
          const index = bullets.indexOf(bullet);
          bullets.splice(index, 1);
        }
      } else {
        // Otherwise, render the bullet and check if it overlaps with the user.
        // If so, end the game.
        paint_cell(bullet, bullet_width, bullet_height, true);
        if (check_invader_bullet_collision_with_user(bullet)) {
          game_over();
          return;
        }
      }
    }
  }

  // Check if the user bullet overlaps with an invader.
  function check_user_bullet_collision_with_invader(bullet) {
    for (var col = num_invader_cols - 1; col >= 0; col--) {
      for (var row = num_invader_rows - 1; row >= 0; row--) {
        const invader = invaders[col][row];
        if (
          invader &&
          Math.floor(bullet.col) == Math.floor(invader.col) &&
          Math.floor(bullet.row) == Math.floor(invader.row)
        ) {
          const index = invaders[col].indexOf(invader);
          invaders[col].splice(index, 1);
          score++;
          return true;
        }
      }
    }
    return false;
  }

  // Check if the bullet overlaps with the user.
  function check_invader_bullet_collision_with_user(bullet) {
    return (
      Math.floor(bullet.col) == Math.floor(user_col) &&
      Math.ceil(bullet.row) == Math.ceil(max_row)
    );
  }

  function invader_potentially_shoot_bullet() {
    // Have a low probability of the invader shooting.
    if (!end_of_game && Math.random() < 0.05) {
      // Randomly pick an invader column.
      const shooting_col_num = Math.floor(Math.random() * num_invader_cols);
      // Shoot from the lowest invader in that column.
      const shooting_col = invaders[shooting_col_num];
      if (shooting_col.length > 0) {
        const lowest_invader = shooting_col[shooting_col.length - 1];
        bullets.push({
          row: lowest_invader.row + 1,
          col: lowest_invader.col + 0.25,
          color: "red",
        });
      }
      return;
    }
  }

  function user_shoot_at_invader() {
    bullets.push({
      col: user_col - 1,
      row: max_row + 0.25,
      color: "blue",
      from_user: true,
    });
  }

  function game_over() {
    end_of_game = true;
    clearInterval(refresh_interval);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas_width, canvas_height);
    ctx.font = "bold 30px Helvetica";
    ctx.fillStyle = "green";
    ctx.fillText("Game Over", 120, 200);
  }

  function start() {
    init_screen();
  }
  start();
});
